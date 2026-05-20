"""
Genesis Savanna Animal Simulation  —  DNA Evolution Edition
=============================================================
サバンナの生態系シミュレーション。
- 地形: 平地・山・森林・水飲み場
- 動物: ライオン, シマウマ, ガゼル, 象
- 行動AI: REINFORCE (種ごとに共有ポリシー)
- DNA進化: 個体ごとに body / behavior 遺伝子を持ち
            エピソードをまたいで世代交代・自然選択する

インストール:
  pip install genesis-world torch onnx onnxruntime numpy

実行:
  python savanna_sim.py
"""

import numpy as np
import torch
import torch.nn as nn
import json
import copy
from dataclasses import dataclass, field, asdict
from typing import List

# ============================================================
# デバイス選択
# ============================================================
def _select_device():
    if torch.cuda.is_available():
        print("[Device] CUDA GPU detected → using cuda")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("[Device] Apple MPS detected → using mps")
        return torch.device("mps")
    else:
        print("[Device] Using CPU")
        return torch.device("cpu")

TRAIN_DEVICE  = _select_device()
GENESIS_DEVICE = torch.device("cpu")

# ============================================================
# Genesis インポート
# ============================================================
try:
    import genesis as gs
    GENESIS_AVAILABLE = True
except ImportError:
    GENESIS_AVAILABLE = False
    print("[WARNING] genesis not found. Running in mock mode.")

# ============================================================
# 世界設定
# ============================================================
WORLD_SIZE = 20
CELL       = 1.0
MAX_STEPS  = 300

# 進化設定
N_GENERATIONS       = 50    # 世代数
EPISODES_PER_GEN    = 10    # 1世代あたりのエピソード数 (平均 fitness 計算用)
ELITE_RATIO         = 0.4   # 上位何割を親として残すか
MUTATION_RATE       = 0.15  # 突然変異確率 (遺伝子ごと)
MUTATION_STRENGTH   = 0.12  # 突然変異の強度 (σ)

# 地形タイプ
TERRAIN_PLAIN    = 0
TERRAIN_MOUNTAIN = 1
TERRAIN_FOREST   = 2
TERRAIN_WATER    = 3

# 動物種
SPECIES_LION    = 0
SPECIES_ZEBRA   = 1
SPECIES_GAZELLE = 2
SPECIES_ELEPHANT= 3
SPECIES_NAMES   = ["Lion", "Zebra", "Gazelle", "Elephant"]

# ============================================================
# DNA — 個体ごとの遺伝子
# ============================================================
@dataclass
class DNA:
    """
    body_genes  : 身体能力 (速度・代謝率・体サイズ)
    behav_genes : 行動傾向 (大胆さ・群れ志向・水場感度)
    fitness     : エピソードをまたいで蓄積する適応度スコア
    """
    # --- 身体能力遺伝子 ---
    speed:       float = 1.0   # 移動速度スケール  (0.5 〜 2.5)
    thirst_rate: float = 0.010 # 渇き増加率        (0.003 〜 0.025)
    hunger_rate: float = 0.008 # 空腹増加率        (0.003 〜 0.020)
    size_scale:  float = 1.0   # 体サイズ倍率      (0.6 〜 1.5)
    # --- 行動傾向遺伝子 ---
    boldness:    float = 0.5   # 捕食者への大胆さ  (0.0 = 臆病 〜 1.0 = 勇敢)
    sociality:   float = 0.5   # 群れ志向          (0.0 = 単独 〜 1.0 = 群れ好き)
    water_sense: float = 0.5   # 水場感知感度      (0.0 〜 1.0, 高いほど水場へ引き寄せ)
    # --- 適応度 (遺伝しない。毎世代リセット) ---
    fitness:     float = 0.0

    # --- 固定情報 (遺伝するが変異しない) ---
    species: int = SPECIES_ZEBRA

    def gene_vector(self) -> np.ndarray:
        """変異・交叉の対象となる遺伝子を配列として返す"""
        return np.array([
            self.speed, self.thirst_rate, self.hunger_rate, self.size_scale,
            self.boldness, self.sociality, self.water_sense,
        ], dtype=np.float64)

    @staticmethod
    def from_gene_vector(vec: np.ndarray, species: int) -> "DNA":
        """配列からDNAを復元"""
        return DNA(
            speed       = float(np.clip(vec[0], 0.5,  2.5)),
            thirst_rate = float(np.clip(vec[1], 0.002, 0.012)),
            hunger_rate = float(np.clip(vec[2], 0.002, 0.010)),
            size_scale  = float(np.clip(vec[3], 0.6,  1.5)),
            boldness    = float(np.clip(vec[4], 0.0,  1.0)),
            sociality   = float(np.clip(vec[5], 0.0,  1.0)),
            water_sense = float(np.clip(vec[6], 0.0,  1.0)),
            species     = species,
        )

    def crossover(self, other: "DNA", rng: np.random.RandomState) -> "DNA":
        """一様交叉: 各遺伝子を50%の確率でどちらの親から受け取るか決める"""
        a = self.gene_vector()
        b = other.gene_vector()
        mask  = rng.rand(len(a)) < 0.5
        child_vec = np.where(mask, a, b)
        return DNA.from_gene_vector(child_vec, self.species)

    def mutate(self, rng: np.random.RandomState) -> "DNA":
        """各遺伝子を MUTATION_RATE の確率でガウスノイズを加えて変異させる"""
        vec = self.gene_vector()
        for i in range(len(vec)):
            if rng.rand() < MUTATION_RATE:
                vec[i] += rng.randn() * MUTATION_STRENGTH
        return DNA.from_gene_vector(vec, self.species)


# ============================================================
# 種ごとの基準パラメータ (DNAのデフォルト値として使う)
# ============================================================
BASE_PARAMS = {
    SPECIES_LION: dict(
        speed=1.5, thirst_rate=0.004, hunger_rate=0.007,
        size_scale=1.0, boldness=0.8, sociality=0.3, water_sense=0.5,
        color=(0.90, 0.70, 0.20, 1.0), base_size=(0.5, 0.5, 0.4),
        can_climb=True,
    ),
    SPECIES_ZEBRA: dict(
        speed=1.3, thirst_rate=0.005, hunger_rate=0.004,
        size_scale=1.0, boldness=0.2, sociality=0.8, water_sense=0.5,
        color=(0.95, 0.95, 0.95, 1.0), base_size=(0.45, 0.45, 0.6),
        can_climb=False,
    ),
    SPECIES_GAZELLE: dict(
        speed=1.8, thirst_rate=0.003, hunger_rate=0.003,
        size_scale=1.0, boldness=0.1, sociality=0.6, water_sense=0.5,
        color=(0.80, 0.60, 0.35, 1.0), base_size=(0.3, 0.3, 0.35),
        can_climb=False,
    ),
    SPECIES_ELEPHANT: dict(
        speed=0.9, thirst_rate=0.008, hunger_rate=0.003,
        size_scale=1.0, boldness=0.6, sociality=0.7, water_sense=0.7,
        color=(0.50, 0.50, 0.55, 1.0), base_size=(0.7, 0.7, 0.8),
        can_climb=False,
    ),
}

N_LIONS     = 3
N_ZEBRAS    = 8
N_GAZELLES  = 10
N_ELEPHANTS = 4
N_ANIMALS   = N_LIONS + N_ZEBRAS + N_GAZELLES + N_ELEPHANTS

AGENT_SPECIES = (
    [SPECIES_LION]     * N_LIONS    +
    [SPECIES_ZEBRA]    * N_ZEBRAS   +
    [SPECIES_GAZELLE]  * N_GAZELLES +
    [SPECIES_ELEPHANT] * N_ELEPHANTS
)

def setup_globals(lions, zebras, gazelles, elephants):
    """argparse の値でグローバル個体数を上書きし AGENT_SPECIES を再構築する"""
    global N_LIONS, N_ZEBRAS, N_GAZELLES, N_ELEPHANTS, N_ANIMALS, AGENT_SPECIES
    N_LIONS     = lions
    N_ZEBRAS    = zebras
    N_GAZELLES  = gazelles
    N_ELEPHANTS = elephants
    N_ANIMALS   = N_LIONS + N_ZEBRAS + N_GAZELLES + N_ELEPHANTS
    AGENT_SPECIES = (
        [SPECIES_LION]     * N_LIONS    +
        [SPECIES_ZEBRA]    * N_ZEBRAS   +
        [SPECIES_GAZELLE]  * N_GAZELLES +
        [SPECIES_ELEPHANT] * N_ELEPHANTS
    )
    print(f"[Config] Lion×{N_LIONS}  Zebra×{N_ZEBRAS}  "
          f"Gazelle×{N_GAZELLES}  Elephant×{N_ELEPHANTS}  "
          f"Total={N_ANIMALS}")

# ============================================================
# 初期個体群の生成
# ============================================================
def make_initial_population(rng: np.random.RandomState) -> List[DNA]:
    """各個体のDNAを基準パラメータ付近でランダム初期化"""
    population = []
    for sp in AGENT_SPECIES:
        bp = BASE_PARAMS[sp]
        dna = DNA(
            speed       = float(np.clip(bp["speed"]       + rng.randn() * 0.15, 0.5,  2.5)),
            thirst_rate = float(np.clip(bp["thirst_rate"] + rng.randn() * 0.001, 0.002, 0.012)),
            hunger_rate = float(np.clip(bp["hunger_rate"] + rng.randn() * 0.001, 0.002, 0.010)),
            size_scale  = float(np.clip(1.0               + rng.randn() * 0.1,   0.6,  1.5)),
            boldness    = float(np.clip(bp["boldness"]    + rng.randn() * 0.1,   0.0,  1.0)),
            sociality   = float(np.clip(bp["sociality"]   + rng.randn() * 0.1,   0.0,  1.0)),
            water_sense = float(np.clip(bp["water_sense"] + rng.randn() * 0.1,   0.0,  1.0)),
            species     = sp,
        )
        population.append(dna)
    return population


# ============================================================
# 自然選択 + 交叉 + 突然変異 → 次世代生成
# ============================================================
def evolve_population(population: List[DNA], rng: np.random.RandomState) -> List[DNA]:
    """
    種ごとに独立して進化させる。
    1. fitness 上位 ELITE_RATIO を親として選択
    2. 親同士の交叉で子を生成
    3. 突然変異を加える
    4. エリート個体はそのまま次世代にも残す (エリート保存)
    """
    next_gen = []

    for sp in range(4):
        # この種の全個体を取り出す
        sp_inds = [dna for dna in population if dna.species == sp]
        n = len(sp_inds)
        if n == 0:
            continue

        # fitness でソート (降順)
        sp_inds.sort(key=lambda d: d.fitness, reverse=True)

        # エリート数 (最低1匹は残す)
        n_elite = max(1, int(n * ELITE_RATIO))
        elites  = sp_inds[:n_elite]

        print(f"  [{SPECIES_NAMES[sp]}] "
              f"best_fitness={elites[0].fitness:.1f}  "
              f"avg_fitness={np.mean([d.fitness for d in sp_inds]):.1f}  "
              f"best_genes: speed={elites[0].speed:.2f} "
              f"thirst={elites[0].thirst_rate:.4f} "
              f"boldness={elites[0].boldness:.2f}")

        # エリートをそのまま保存 (fitnessはリセット)
        for e in elites:
            child = copy.deepcopy(e)
            child.fitness = 0.0
            next_gen.append(child)

        # 残りの枠を交叉 + 突然変異で埋める
        while len([d for d in next_gen if d.species == sp]) < n:
            # 親を fitness に比例した確率でサンプル (トーナメント選択)
            p1 = _tournament_select(elites, rng)
            p2 = _tournament_select(elites, rng)
            child = p1.crossover(p2, rng).mutate(rng)
            child.fitness = 0.0
            next_gen.append(child)

    return next_gen


def _tournament_select(candidates: List[DNA], rng: np.random.RandomState,
                        k: int = 2) -> DNA:
    """k匹をランダムに選び、最も fitness が高い個体を返す"""
    chosen = rng.choice(len(candidates), size=min(k, len(candidates)), replace=False)
    return max([candidates[i] for i in chosen], key=lambda d: d.fitness)


# ============================================================
# 地形マップ
# ============================================================
def make_terrain_map():
    grid = np.full((WORLD_SIZE, WORLD_SIZE), TERRAIN_PLAIN, dtype=int)
    mountains = [(3,15),(3,16),(4,15),(4,16),(4,17),(5,15),(5,16),
                 (2,14),(3,14),(15,3),(16,3),(16,4),(17,4)]
    forests   = [(8,2),(8,3),(9,2),(9,3),(9,4),(10,2),(10,3),
                 (11,1),(11,2),(12,2),(12,3),(6,1),(7,1),(7,2)]
    waters    = [(10,10),(10,11),(11,10),(11,11),(16,16),(16,17),(17,16)]
    for r,c in mountains: grid[r][c] = TERRAIN_MOUNTAIN
    for r,c in forests:   grid[r][c] = TERRAIN_FOREST
    for r,c in waters:    grid[r][c] = TERRAIN_WATER
    return grid

TERRAIN_MAP = make_terrain_map()

TERRAIN_HEIGHT = {
    TERRAIN_PLAIN: 0.05, TERRAIN_MOUNTAIN: 3.0,
    TERRAIN_FOREST: 0.3, TERRAIN_WATER: 0.01,
}
TERRAIN_COLORS = {
    TERRAIN_PLAIN:    (0.82, 0.72, 0.45, 1.0),
    TERRAIN_MOUNTAIN: (0.60, 0.55, 0.50, 1.0),
    TERRAIN_FOREST:   (0.25, 0.55, 0.25, 1.0),
    TERRAIN_WATER:    (0.20, 0.55, 0.85, 1.0),
}

# ============================================================
# Genesis シーン構築
# ============================================================
def build_genesis_scene(population: List[DNA]):
    gs.init(precision="32", logging_level="warning", backend=gs.cpu)
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(10.0, -6.0, 28.0),
            camera_lookat=(10.0, 10.0, 0.0),
            camera_fov=55, max_FPS=60,
        ),
        show_viewer=True, show_FPS=True,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Default(color=(0.75, 0.65, 0.40, 1.0)),
    )

    for r in range(WORLD_SIZE):
        for c in range(WORLD_SIZE):
            t = TERRAIN_MAP[r][c]
            if t == TERRAIN_PLAIN:
                continue
            h     = TERRAIN_HEIGHT[t]
            color = TERRAIN_COLORS[t]
            cx    = c * CELL + CELL / 2
            cy    = r * CELL + CELL / 2

            if t == TERRAIN_MOUNTAIN:
                scene.add_entity(
                    morph=gs.morphs.Box(pos=(cx,cy,h/2), size=(CELL*.85,CELL*.85,h), fixed=True),
                    surface=gs.surfaces.Default(color=color))
                scene.add_entity(
                    morph=gs.morphs.Box(pos=(cx,cy,h+.25), size=(CELL*.35,CELL*.35,.5), fixed=True),
                    surface=gs.surfaces.Default(color=(.55,.52,.50,1.0)))
            elif t == TERRAIN_FOREST:
                scene.add_entity(
                    morph=gs.morphs.Box(pos=(cx,cy,.4), size=(.12,.12,.8), fixed=True),
                    surface=gs.surfaces.Default(color=(.45,.28,.10,1.0)))
                scene.add_entity(
                    morph=gs.morphs.Box(pos=(cx,cy,1.1), size=(.6,.6,.7), fixed=True),
                    surface=gs.surfaces.Default(color=color))
            elif t == TERRAIN_WATER:
                scene.add_entity(
                    morph=gs.morphs.Box(pos=(cx,cy,.02), size=(CELL*.95,CELL*.95,.04), fixed=True),
                    surface=gs.surfaces.Default(color=color))

    for i in range(0, WORLD_SIZE+1, 5):
        for axis in [(WORLD_SIZE*CELL/2, i*CELL, WORLD_SIZE*CELL, .03),
                     (i*CELL, WORLD_SIZE*CELL/2, .03, WORLD_SIZE*CELL)]:
            scene.add_entity(
                morph=gs.morphs.Box(pos=(axis[0],axis[1],.002),
                                    size=(axis[2],axis[3],.004), fixed=True),
                surface=gs.surfaces.Default(color=(.6,.6,.5,.3)))

    animals = []
    spawn   = _make_spawn_positions()
    for i, dna in enumerate(population):
        sp     = dna.species
        bp     = BASE_PARAMS[sp]
        bsz    = bp["base_size"]
        sz     = tuple(s * dna.size_scale for s in bsz)
        rx, ry = spawn[i]
        pz     = sz[2] / 2 + 0.05
        a = scene.add_entity(
            material=gs.materials.Rigid(rho=500.0, friction=0.5),
            morph=gs.morphs.Box(pos=(rx, ry, pz), size=sz),
            surface=gs.surfaces.Default(color=bp["color"]),
        )
        animals.append(a)

    scene.build()
    return scene, animals


def _make_spawn_positions():
    rng = np.random.RandomState(7)
    pos = []
    # ライオンは東側に配置 (草食獣エリアと少し離す)
    for _ in range(N_LIONS):
        pos.append((rng.uniform(13, 19), rng.uniform(4, 16)))
    # シマウマは中央平地に広く配置
    for _ in range(N_ZEBRAS):
        pos.append((rng.uniform(3, 17),  rng.uniform(3, 17)))
    # ガゼルは西〜南西 (森林近く)
    for _ in range(N_GAZELLES):
        pos.append((rng.uniform(2, 14),  rng.uniform(1, 10)))
    # 象は水場の近く (中央〜南東)
    for _ in range(N_ELEPHANTS):
        pos.append((rng.uniform(7, 14),  rng.uniform(7, 14)))
    return pos


# ============================================================
# ポリシーネットワーク (種ごとに共有)
# ============================================================
OBS_DIM = 16   # +2: boldness, sociality を観測に追加
ACT_DIM = 5

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 128), nn.ReLU(),
            nn.Linear(128, 128),     nn.ReLU(),
            nn.Linear(128, ACT_DIM),
        )
    def forward(self, x):
        return self.net(x)


# ============================================================
# サバンナシミュレーター (DNA対応)
# ============================================================
MOVES = [(0,1),(0,-1),(-1,0),(1,0),(0,0)]

class SavannaSimulator:
    # 水場座標を起動時に1回だけ計算してクラス変数にキャッシュ
    _WATER_POSITIONS = np.array(
        [[c + 0.5, r + 0.5]
         for r in range(WORLD_SIZE)
         for c in range(WORLD_SIZE)
         if TERRAIN_MAP[r][c] == TERRAIN_WATER],
        dtype=np.float32
    )
    def __init__(self, population: List[DNA]):
        self.terrain    = TERRAIN_MAP
        self.population = population   # 個体ごとのDNA
        self.reset()

    def reset(self):
        spawn           = _make_spawn_positions()
        self.positions  = np.array(spawn, dtype=float)
        self.thirst     = np.zeros(N_ANIMALS)
        self.hunger     = np.zeros(N_ANIMALS)
        self.alive      = np.ones(N_ANIMALS, dtype=bool)
        self.step_count = 0
        # fitness計算用カウンタ
        self.survived_steps = np.zeros(N_ANIMALS)
        self.kills          = np.zeros(N_ANIMALS)
        return [self._obs(i) for i in range(N_ANIMALS)]

    def _terrain_at(self, pos):
        r = int(np.clip(pos[1], 0, WORLD_SIZE-1))
        c = int(np.clip(pos[0], 0, WORLD_SIZE-1))
        return self.terrain[r][c]

    def _nearest(self, aid, species_filter=None, exclude_self=True):
        # alive マスク
        mask = self.alive.copy()
        if exclude_self:
            mask[aid] = False
        if species_filter is not None:
            for i in range(N_ANIMALS):
                if AGENT_SPECIES[i] != species_filter:
                    mask[i] = False
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return np.zeros(2), float(WORLD_SIZE)
        diffs = self.positions[indices] - self.positions[aid]
        dists = np.linalg.norm(diffs, axis=1)
        idx   = np.argmin(dists)
        return diffs[idx], float(dists[idx])

    def _nearest_water(self, pos):
        # キャッシュ済み水場座標に対してベクトル演算 (400ループ → 1回のnp演算)
        diffs = self._WATER_POSITIONS - pos
        dists = np.linalg.norm(diffs, axis=1)
        idx   = np.argmin(dists)
        return diffs[idx], float(dists[idx])

    def _obs(self, aid):
        """
        観測ベクトル (16次元):
          0,1  : 位置
          2    : 地形タイプ
          3    : 渇き
          4    : 空腹
          5,6,7: 水場方向・距離
          8,9,10: 脅威方向・距離
          11   : 森林近傍
          12,13: 同種仲間方向
          14   : boldness  (DNA遺伝子をそのまま観測に渡す)
          15   : water_sense
        """
        pos = self.positions[aid]
        sp  = AGENT_SPECIES[aid]
        dna = self.population[aid]
        t   = self._terrain_at(pos)

        water_diff, water_dist = self._nearest_water(pos)

        if sp == SPECIES_LION:
            threat_diff, threat_dist = np.zeros(2), WORLD_SIZE
            for ts in [SPECIES_ZEBRA, SPECIES_GAZELLE, SPECIES_ELEPHANT]:
                d, dist = self._nearest(aid, species_filter=ts)
                if dist < threat_dist:
                    threat_dist = dist
                    threat_diff = d
        else:
            threat_diff, threat_dist = self._nearest(aid, species_filter=SPECIES_LION)

        forest_near = 1.0 if t == TERRAIN_FOREST else 0.0
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr = int(np.clip(pos[1]+dr, 0, WORLD_SIZE-1))
            nc = int(np.clip(pos[0]+dc, 0, WORLD_SIZE-1))
            if self.terrain[nr][nc] == TERRAIN_FOREST:
                forest_near = 1.0
                break

        conspecific_diff, _ = self._nearest(aid, species_filter=sp)

        obs = np.array([
            pos[0]/WORLD_SIZE, pos[1]/WORLD_SIZE,
            t/3.0,
            float(self.thirst[aid]),
            float(self.hunger[aid]),
            water_diff[0]/WORLD_SIZE, water_diff[1]/WORLD_SIZE,
            min(water_dist/WORLD_SIZE, 1.0),
            threat_diff[0]/WORLD_SIZE, threat_diff[1]/WORLD_SIZE,
            min(threat_dist/WORLD_SIZE, 1.0),
            forest_near,
            conspecific_diff[0]/WORLD_SIZE, conspecific_diff[1]/WORLD_SIZE,
            dna.boldness,      # DNA遺伝子を直接観測へ
            dna.water_sense,
        ], dtype=np.float32)
        obs = np.clip(obs, -1.0, 1.0)
        # NaN/inf ガード: 万一混入していたらゼロに置換
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def step(self, aid, action):
        if not self.alive[aid]:
            return self._obs(aid), 0.0, True

        sp     = AGENT_SPECIES[aid]
        dna    = self.population[aid]
        bp     = BASE_PARAMS[sp]
        dx, dy = MOVES[action]

        # DNA の speed を使って移動距離を決める
        new_pos = self.positions[aid] + np.array([dx, dy]) * dna.speed * 0.3
        new_pos = np.clip(new_pos, 0, WORLD_SIZE - 0.01)

        new_terrain = self._terrain_at(new_pos)
        if new_terrain == TERRAIN_MOUNTAIN and not bp["can_climb"]:
            new_pos     = self.positions[aid]
            new_terrain = self._terrain_at(new_pos)

        self.positions[aid] = new_pos

        # DNA の代謝率を使って生理値を更新
        self.thirst[aid] += dna.thirst_rate
        self.hunger[aid] += dna.hunger_rate

        # 水場回復: water_sense が高いほど回復量UP
        if new_terrain == TERRAIN_WATER:
            recover = 0.25 + dna.water_sense * 0.25   # 0.25 〜 0.50
            self.thirst[aid] = max(0.0, self.thirst[aid] - recover)

        # 草食獣の採食
        if sp != SPECIES_LION and new_terrain in [TERRAIN_PLAIN, TERRAIN_FOREST]:
            self.hunger[aid] = max(0.0, self.hunger[aid] - 0.05)

        reward = self._compute_reward(aid, new_pos, new_terrain)

        # 捕食チェック: ライオンの size_scale が大きいほど捕食半径UP
        if sp == SPECIES_LION:
            catch_radius = 0.6 + dna.size_scale * 0.25
            prey_mask = np.array([
                self.alive[i] and AGENT_SPECIES[i] != SPECIES_LION
                for i in range(N_ANIMALS)
            ])
            prey_ids = np.where(prey_mask)[0]
            if len(prey_ids) > 0:
                dists = np.linalg.norm(self.positions[prey_ids] - new_pos, axis=1)
                caught = prey_ids[dists < catch_radius]
                for prey_id in caught:
                    self.alive[prey_id]  = False
                    self.hunger[aid]     = max(0.0, self.hunger[aid] - 0.5)
                    reward              += 10.0
                    self.kills[aid]     += 1

        # 死亡チェック
        if self.thirst[aid] >= 1.0 or self.hunger[aid] >= 1.0:
            self.alive[aid] = False
            reward         -= 5.0
        else:
            self.survived_steps[aid] += 1

        self.step_count += 1
        done = self.step_count >= MAX_STEPS or not self.alive.any()
        return self._obs(aid), reward, done

    def _compute_reward(self, aid, pos, terrain):
        sp  = AGENT_SPECIES[aid]
        dna = self.population[aid]
        r   = -0.01
        r  -= self.thirst[aid] * 0.5
        r  -= self.hunger[aid] * 0.3

        if sp == SPECIES_LION:
            # 生きている草食獣との距離を一括計算
            prey_mask = np.array([
                self.alive[i] and AGENT_SPECIES[i] != SPECIES_LION
                for i in range(N_ANIMALS)
            ])
            if prey_mask.any():
                prey_pos = self.positions[prey_mask]
                dists    = np.linalg.norm(prey_pos - pos, axis=1)
                close    = dists[dists < 3.0]
                r       += float(np.sum(3.0 - close) * 0.5)
            _, wd = self._nearest_water(pos)
            if wd < 1.5: r += 1.0
        else:
            _, lion_dist = self._nearest(aid, species_filter=SPECIES_LION)
            # boldness が低いほどライオン接近ペナルティが大きくなる
            if lion_dist < 4.0:
                fear_scale = 1.0 + (1.0 - dna.boldness)
                r -= (4.0 - lion_dist) * fear_scale
            if terrain == TERRAIN_FOREST:
                r += 0.8 if lion_dist < 5.0 else 0.3
            _, wd = self._nearest_water(pos)
            # water_sense が高いほど水場ボーナスが大きくなる
            if wd < 1.5:
                r += 1.0 + dna.water_sense * 1.0

        return r

    def compute_fitness(self) -> None:
        """
        エピソード終了時に各個体のDNA.fitnessへ加算する。
        fitness = 生存ステップ数 + 捕食数×15 - (渇き+空腹)×10
        """
        for i in range(N_ANIMALS):
            dna = self.population[i]
            penalty = (self.thirst[i] + self.hunger[i]) * 10.0
            dna.fitness += (
                self.survived_steps[i]
                + self.kills[i] * 15.0
                - penalty
            )


# ============================================================
# Genesis エージェント位置更新
# ============================================================
def update_genesis_animals(animals, sim):
    for i, animal in enumerate(animals):
        if not sim.alive[i]:
            animal.set_pos(torch.tensor([[0, 0, -10.0]], device='cpu'))
            continue
        dna = sim.population[i]
        sp  = dna.species
        bp  = BASE_PARAMS[sp]
        px  = float(sim.positions[i][0])
        py  = float(sim.positions[i][1])
        bsz = bp["base_size"]
        pz  = bsz[2] * dna.size_scale / 2 + 0.05
        t   = sim._terrain_at(sim.positions[i])
        if t == TERRAIN_MOUNTAIN: pz += TERRAIN_HEIGHT[TERRAIN_MOUNTAIN]
        elif t == TERRAIN_FOREST: pz += 0.2
        animal.set_pos(torch.tensor([[px, py, pz]], device='cpu'))


# ============================================================
# 学習ループ (世代 × エピソード)
# ============================================================
def train(n_generations=None, episodes_per_gen=None):
    _gens = n_generations    if n_generations    is not None else N_GENERATIONS
    _eps  = episodes_per_gen if episodes_per_gen is not None else EPISODES_PER_GEN

    print("=" * 65)
    print("Genesis Savanna  —  DNA Evolution + REINFORCE")
    print(f"  Animals: Lion×{N_LIONS} Zebra×{N_ZEBRAS} "
          f"Gazelle×{N_GAZELLES} Elephant×{N_ELEPHANTS} (total={N_ANIMALS})")
    print(f"  {_gens} generations × {_eps} episodes/gen")
    print(f"  Elite ratio: {ELITE_RATIO}  Mutation rate: {MUTATION_RATE}")
    print("=" * 65)

    rng = np.random.RandomState(0)

    # 初期個体群
    population = make_initial_population(rng)

    # 種ごとのポリシー (世代をまたいで継続学習)
    policies   = {sp: PolicyNet().to(TRAIN_DEVICE) for sp in range(4)}
    optimizers = {sp: torch.optim.Adam(policies[sp].parameters(), lr=8e-4)
                  for sp in range(4)}

    scene, genesis_animals = None, None
    if GENESIS_AVAILABLE:
        print("[Genesis] Building scene...")
        scene, genesis_animals = build_genesis_scene(population)
        print("[Genesis] Ready.\n")

    history = []   # 世代ごとの統計を記録

    for gen in range(_gens):
        print(f"\n{'='*20} Generation {gen+1}/{_gens} {'='*20}")

        # fitness をリセット
        for dna in population:
            dna.fitness = 0.0

        sim = SavannaSimulator(population)

        # ── 複数エピソード走らせて fitness を蓄積 ──────────────────
        for ep in range(_eps):
            obs_all       = sim.reset()
            log_probs_all = {i: [] for i in range(N_ANIMALS)}
            rewards_all   = {i: [] for i in range(N_ANIMALS)}

            for step in range(MAX_STEPS):
                if not sim.alive.any(): break

                for i in range(N_ANIMALS):
                    if not sim.alive[i]: continue
                    sp  = AGENT_SPECIES[i]
                    obs = torch.FloatTensor(obs_all[i]).unsqueeze(0).to(TRAIN_DEVICE)
                    logits = policies[sp](obs)
                    probs  = torch.softmax(logits, dim=-1)
                    dist   = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    log_probs_all[i].append(dist.log_prob(action))
                    obs_all[i], reward, done = sim.step(i, action.item())
                    rewards_all[i].append(reward)

                if scene is not None and step % 3 == 0:
                    try:
                        update_genesis_animals(genesis_animals, sim)
                        scene.step()
                    except Exception:
                        pass

            # fitness 加算
            sim.compute_fitness()

            # REINFORCE 更新
            for sp in range(4):
                sp_indices = [i for i in range(N_ANIMALS) if AGENT_SPECIES[i] == sp]
                total_loss = torch.tensor(0.0, device=TRAIN_DEVICE)
                for i in sp_indices:
                    if len(rewards_all[i]) == 0: continue
                    G, returns = 0.0, []
                    for r in reversed(rewards_all[i]):
                        G = r + 0.99 * G
                        returns.insert(0, G)
                    R  = torch.tensor(returns, dtype=torch.float32, device=TRAIN_DEVICE)
                    # std がゼロのエピソード (全報酬が同値) では正規化しない
                    if R.std() > 1e-8:
                        R = (R - R.mean()) / R.std()
                    else:
                        R = R - R.mean()
                    lp = torch.stack(log_probs_all[i])
                    total_loss = total_loss + (-(lp * R).sum())
                if total_loss.item() != 0:
                    optimizers[sp].zero_grad()
                    total_loss.backward()
                    # 勾配クリッピング: 学習が不安定になるのを防ぐ
                    torch.nn.utils.clip_grad_norm_(policies[sp].parameters(), max_norm=1.0)
                    optimizers[sp].step()

        # ── 世代統計を表示 ─────────────────────────────────────────
        gen_stats = {}
        for sp in range(4):
            sp_dnas = [d for d in population if d.species == sp]
            fitnesses = [d.fitness for d in sp_dnas]
            gen_stats[sp] = {
                "best":   max(fitnesses),
                "avg":    np.mean(fitnesses),
                "speed":  np.mean([d.speed for d in sp_dnas]),
                "thirst": np.mean([d.thirst_rate for d in sp_dnas]),
                "bold":   np.mean([d.boldness for d in sp_dnas]),
                "w_sens": np.mean([d.water_sense for d in sp_dnas]),
            }
        history.append(gen_stats)

        print("  Fitness & evolved traits:")
        for sp in range(4):
            s = gen_stats[sp]
            print(f"    {SPECIES_NAMES[sp]:8s}: "
                  f"best={s['best']:6.0f}  avg={s['avg']:6.0f}  "
                  f"speed={s['speed']:.2f}  "
                  f"thirst={s['thirst']:.4f}  "
                  f"boldness={s['bold']:.2f}  "
                  f"water_sense={s['w_sens']:.2f}")

        # ── 自然選択 → 次世代生成 ──────────────────────────────────
        population = evolve_population(population, rng)

        # Genesis の個体を新世代の size_scale に合わせて更新
        if scene is not None:
            sim_new = SavannaSimulator(population)
            try:
                update_genesis_animals(genesis_animals, sim_new)
            except Exception:
                pass

    print("\n[Done] Evolution complete.")

    # 進化ログを保存
    _save_history(history)
    return policies, population, history


def _save_history(history):
    serializable = []
    for gen_stats in history:
        row = {}
        for sp, stats in gen_stats.items():
            row[SPECIES_NAMES[sp]] = {k: float(v) for k, v in stats.items()}
        serializable.append(row)
    with open("evolution_history.json", "w") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print("[JSON] Saved: evolution_history.json")


# ============================================================
# ONNX エクスポート
# ============================================================
def export_onnx(policies, prefix="savanna_policy"):
    for sp, policy in policies.items():
        path = f"{prefix}_{SPECIES_NAMES[sp].lower()}.onnx"
        policy.eval().to('cpu')
        dummy = torch.zeros(1, OBS_DIM, device='cpu')
        torch.onnx.export(
            policy, dummy, path,
            input_names=["state"], output_names=["logits"],
            dynamic_axes={"state": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=11,
        )
        print(f"[ONNX] Saved: {path}")


# ============================================================
# エントリポイント
# ============================================================
if __name__ == "__main__":
    import argparse
    import signal

    parser = argparse.ArgumentParser(description="Genesis Savanna DNA Evolution")
    parser.add_argument("--lions",     type=int, default=3,  help="ライオン数 (default: 3)")
    parser.add_argument("--zebras",    type=int, default=8,  help="シマウマ数 (default: 8)")
    parser.add_argument("--gazelles",  type=int, default=10, help="ガゼル数   (default: 10)")
    parser.add_argument("--elephants", type=int, default=4,  help="象数       (default: 4)")
    parser.add_argument("--gens",      type=int, default=50, help="世代数     (default: 50)")
    parser.add_argument("--eps",       type=int, default=10, help="世代あたりエピソード数 (default: 10)")
    args = parser.parse_args()

    # グローバル個体数を引数で上書き
    setup_globals(args.lions, args.zebras, args.gazelles, args.elephants)
    N_GENERATIONS    = args.gens
    EPISODES_PER_GEN = args.eps

    # Ctrl+C で即終了
    def _force_exit(sig, frame):
        print("\n[STOP] Ctrl+C detected — force exit.")
        import os; os._exit(0)

    signal.signal(signal.SIGINT,  _force_exit)
    signal.signal(signal.SIGTERM, _force_exit)

    try:
        policies, final_population, history = train(
            n_generations=args.gens,
            episodes_per_gen=args.eps,
        )
        export_onnx(policies)

        print("\n── Final generation gene summary ──")
        for sp in range(4):
            candidates = [d for d in final_population if d.species == sp]
            if not candidates:
                continue
            best = max(candidates, key=lambda d: d.fitness)
            print(f"  {SPECIES_NAMES[sp]:8s}: "
                  f"speed={best.speed:.2f}  "
                  f"thirst={best.thirst_rate:.4f}  "
                  f"hunger={best.hunger_rate:.4f}  "
                  f"boldness={best.boldness:.2f}  "
                  f"water_sense={best.water_sense:.2f}")

    except KeyboardInterrupt:
        print("\n[STOP] Training interrupted.")
        import os; os._exit(0)
