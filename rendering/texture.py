"""
游戏内程序化纹理生成与材质系统 - 核心模块

功能：
- generate_texture(): 基于算法的程序化纹理生成（噪声/棋盘格/砖块/木纹/大理石），参数可调
- material_synthesis(): 多层材质合成，支持混合模式与粗糙度/金属度/法线属性
- noise_function(): Perlin 噪声与分形噪声（fBm）
- texture_mapping(): UV 纹理映射，支持最近邻/双线性过滤与多种环绕模式
- texture_atlas(): 纹理图集打包与坐标查询
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import random
import time

Color = Tuple[int, int, int]
Texture = List[List[Color]]


@dataclass
class Config:
    pass


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _fade(t: float) -> float:
    return t * t * t * (t * (t * 6 - 15) + 10)


def _grad(hash_value: int, x: float, y: float) -> float:
    h = hash_value & 7
    u = x if h < 4 else y
    v = y if h < 4 else x
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


def _validate_texture(texture: Texture) -> Tuple[int, int]:
    if not texture or not texture[0]:
        raise ValueError("纹理不能为空")
    return len(texture[0]), len(texture)


class ProceduralTexture:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []
        self._perm_cache: Dict[int, List[int]] = {}

    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []

    # ------------------------------------------------------------------
    # 噪声函数：Perlin 噪声 + 分形噪声
    # ------------------------------------------------------------------
    def _permutation(self, seed: int) -> List[int]:
        if seed not in self._perm_cache:
            perm = list(range(256))
            random.Random(seed).shuffle(perm)
            self._perm_cache[seed] = perm * 2
        return self._perm_cache[seed]

    def _perlin(self, x: float, y: float, perm: List[int]) -> float:
        xi = math.floor(x) & 255
        yi = math.floor(y) & 255
        xf = x - math.floor(x)
        yf = y - math.floor(y)
        u = _fade(xf)
        v = _fade(yf)
        aa = perm[perm[xi] + yi]
        ab = perm[perm[xi] + yi + 1]
        ba = perm[perm[xi + 1] + yi]
        bb = perm[perm[xi + 1] + yi + 1]
        x1 = _lerp(_grad(aa, xf, yf), _grad(ba, xf - 1, yf), u)
        x2 = _lerp(_grad(ab, xf, yf - 1), _grad(bb, xf - 1, yf - 1), u)
        return _clamp(_lerp(x1, x2, v), -1.0, 1.0)

    def noise_function(self, x: float, y: float, noise_type: str = "perlin",
                       seed: int = 0, octaves: int = 4,
                       persistence: float = 0.5, lacunarity: float = 2.0,
                       frequency: float = 1.0) -> float:
        """生成平滑噪声，返回值范围 [-1, 1]。

        noise_type: "perlin" 单层平滑噪声；"fractal" 分形噪声（fBm，细节更丰富）。
        """
        if octaves < 1:
            raise ValueError("octaves 必须 >= 1")
        perm = self._permutation(seed)
        if noise_type == "perlin":
            return self._perlin(x * frequency, y * frequency, perm)
        if noise_type == "fractal":
            total = 0.0
            amplitude = 1.0
            freq = frequency
            max_amplitude = 0.0
            for _ in range(octaves):
                total += self._perlin(x * freq, y * freq, perm) * amplitude
                max_amplitude += amplitude
                amplitude *= persistence
                freq *= lacunarity
            return _clamp(total / max_amplitude, -1.0, 1.0)
        raise ValueError(f"未知噪声类型: {noise_type}")

    # ------------------------------------------------------------------
    # 程序化纹理生成：算法 + 可调参数
    # ------------------------------------------------------------------
    def generate_texture(self, width: int, height: int,
                         algorithm: str = "noise",
                         params: Optional[Dict] = None) -> Texture:
        """按算法程序化生成纹理，返回 height x width 的 (r, g, b) 像素阵列。"""
        if width <= 0 or height <= 0:
            raise ValueError("纹理尺寸必须为正数")
        params = dict(params or {})
        seed = params.get("seed", 0)
        color_a = params.get("color_a", (32, 32, 32))
        color_b = params.get("color_b", (224, 224, 224))

        def shade(t: float) -> Color:
            t = _clamp(t, 0.0, 1.0)
            return tuple(int(round(_lerp(a, b, t))) for a, b in zip(color_a, color_b))

        texture: Texture = []
        for y in range(height):
            row: List[Color] = []
            for x in range(width):
                if algorithm == "noise":
                    scale = params.get("scale", 4.0)
                    octaves = params.get("octaves", 4)
                    n = self.noise_function(x / width * scale, y / height * scale,
                                            noise_type="fractal", seed=seed,
                                            octaves=octaves)
                    row.append(shade((n + 1.0) * 0.5))
                elif algorithm == "checker":
                    cell = max(1, params.get("cell_size", max(1, width // 8)))
                    row.append(color_a if (x // cell + y // cell) % 2 == 0 else color_b)
                elif algorithm == "brick":
                    brick_w = max(1, params.get("brick_width", max(1, width // 4)))
                    brick_h = max(1, params.get("brick_height", max(1, height // 8)))
                    mortar = params.get("mortar_color", color_b)
                    offset = (brick_w // 2) if (y // brick_h) % 2 else 0
                    in_mortar = (y % brick_h == 0) or ((x + offset) % brick_w == 0)
                    row.append(mortar if in_mortar else color_a)
                elif algorithm == "wood":
                    rings = params.get("rings", 8.0)
                    turbulence = params.get("turbulence", 4.0)
                    dx = x / width - 0.5
                    dy = y / height - 0.5
                    dist = math.sqrt(dx * dx + dy * dy)
                    n = self.noise_function(x / width * 4.0, y / height * 4.0,
                                            noise_type="fractal", seed=seed)
                    t = 0.5 * (1.0 + math.sin((dist * rings + n * turbulence * 0.1) * 2 * math.pi))
                    row.append(shade(t))
                elif algorithm == "marble":
                    veins = params.get("veins", 5.0)
                    turbulence = params.get("turbulence", 3.0)
                    n = self.noise_function(x / width * 4.0, y / height * 4.0,
                                            noise_type="fractal", seed=seed)
                    t = 0.5 * (1.0 + math.sin((x / width * veins + n * turbulence) * math.pi))
                    row.append(shade(t))
                else:
                    raise ValueError(f"未知纹理算法: {algorithm}")
            texture.append(row)
        self._history.append(("generate_texture", algorithm, width, height))
        return texture

    # ------------------------------------------------------------------
    # 材质合成：多层叠加 + 粗糙度/金属度/法线属性
    # ------------------------------------------------------------------
    @staticmethod
    def _blend_channel(dst: float, src: float, mode: str) -> float:
        if mode == "normal":
            return src
        if mode == "multiply":
            return dst * src / 255.0
        if mode == "add":
            return min(255.0, dst + src)
        if mode == "overlay":
            if dst < 128:
                return 2 * dst * src / 255.0
            return 255.0 - 2 * (255.0 - dst) * (255.0 - src) / 255.0
        raise ValueError(f"未知混合模式: {mode}")

    def material_synthesis(self, layers: List[Dict],
                           properties: Optional[Dict] = None) -> Dict:
        """将多个材质层自下而上合成，并附加粗糙度/金属度/法线属性。

        每层: {"texture": Texture} 或 {"color": (r,g,b)}，
        可选 "opacity" (0~1) 与 "blend_mode" (normal/multiply/add/overlay)。
        """
        if not layers:
            raise ValueError("至少需要一个材质层")
        properties = dict(properties or {})

        size = None
        for layer in layers:
            if "texture" in layer and layer["texture"] is not None:
                size = _validate_texture(layer["texture"])
                break
        if size is None:
            if "size" not in properties:
                raise ValueError("纯颜色层材质必须通过 properties['size'] 指定尺寸")
            size = tuple(properties["size"])
        width, height = size

        def layer_pixel(layer: Dict, x: int, y: int) -> Color:
            if "texture" in layer and layer["texture"] is not None:
                tex_w, tex_h = _validate_texture(layer["texture"])
                return layer["texture"][y % tex_h][x % tex_w]
            if "color" in layer:
                return tuple(layer["color"])
            raise ValueError("材质层必须包含 'texture' 或 'color'")

        result: List[List[Tuple[float, float, float]]] = [
            [tuple(float(c) for c in layer_pixel(layers[0], x, y))
             for x in range(width)]
            for y in range(height)
        ]
        for layer in layers[1:]:
            opacity = _clamp(layer.get("opacity", 1.0), 0.0, 1.0)
            mode = layer.get("blend_mode", "normal")
            for y in range(height):
                for x in range(width):
                    src = layer_pixel(layer, x, y)
                    dst = result[y][x]
                    result[y][x] = tuple(
                        _lerp(d, self._blend_channel(d, s, mode), opacity)
                        for d, s in zip(dst, src)
                    )

        color_tex: Texture = [
            [tuple(int(round(_clamp(c, 0.0, 255.0))) for c in pixel)
             for pixel in row]
            for row in result
        ]
        roughness = _clamp(float(properties.get("roughness", 0.5)), 0.0, 1.0)
        metallic = _clamp(float(properties.get("metallic", 0.0)), 0.0, 1.0)
        material = {
            "color": color_tex,
            "roughness": roughness,
            "metallic": metallic,
            "normal": properties.get("normal"),
            "size": (width, height),
            "layers": len(layers),
        }
        self._history.append(("material_synthesis", len(layers)))
        return material

    # ------------------------------------------------------------------
    # 纹理映射：UV 坐标 + 最近邻/双线性过滤
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_coord(t: float, wrap: str) -> float:
        if wrap == "repeat":
            return t % 1.0
        if wrap == "clamp":
            return _clamp(t, 0.0, 1.0)
        if wrap == "mirror":
            t = t % 2.0
            return t if t <= 1.0 else 2.0 - t
        raise ValueError(f"未知环绕模式: {wrap}")

    @staticmethod
    def _wrap_index(i: int, n: int, wrap: str) -> int:
        if wrap == "repeat":
            return i % n
        if wrap == "clamp":
            return int(_clamp(i, 0, n - 1))
        if wrap == "mirror":
            period = 2 * n
            i %= period
            return i if i < n else period - 1 - i
        raise ValueError(f"未知环绕模式: {wrap}")

    def texture_mapping(self, texture: Texture, u: float, v: float,
                        filter_mode: str = "bilinear",
                        wrap: str = "repeat"):
        """按 UV 坐标采样纹理。

        filter_mode: "nearest" 最近邻；"bilinear" 双线性插值（放大不再马赛克）。
        """
        width, height = _validate_texture(texture)
        u = self._wrap_coord(u, wrap)
        v = self._wrap_coord(v, wrap)

        if filter_mode == "nearest":
            x = min(int(u * width), width - 1)
            y = min(int(v * height), height - 1)
            return texture[y][x]
        if filter_mode == "bilinear":
            sx = u * width - 0.5
            sy = v * height - 0.5
            x0 = math.floor(sx)
            y0 = math.floor(sy)
            fx = sx - x0
            fy = sy - y0
            c00 = texture[self._wrap_index(y0, height, wrap)][self._wrap_index(x0, width, wrap)]
            c10 = texture[self._wrap_index(y0, height, wrap)][self._wrap_index(x0 + 1, width, wrap)]
            c01 = texture[self._wrap_index(y0 + 1, height, wrap)][self._wrap_index(x0, width, wrap)]
            c11 = texture[self._wrap_index(y0 + 1, height, wrap)][self._wrap_index(x0 + 1, width, wrap)]
            return tuple(
                _lerp(_lerp(c00[i], c10[i], fx), _lerp(c01[i], c11[i], fx), fy)
                for i in range(3)
            )
        raise ValueError(f"未知过滤模式: {filter_mode}")

    # ------------------------------------------------------------------
    # 纹理图集：打包 + 坐标
    # ------------------------------------------------------------------
    def texture_atlas(self, textures: Dict[str, Texture],
                      padding: int = 0) -> Dict:
        """将多个纹理打包到一张图集，返回图集纹理与每个纹理的坐标/UV。"""
        if not textures:
            raise ValueError("纹理集合不能为空")
        items = []
        total_area = 0
        for name, tex in textures.items():
            w, h = _validate_texture(tex)
            items.append((name, tex, w, h))
            total_area += w * h
        items.sort(key=lambda item: (-item[3], -item[2]))

        shelf_limit = max(math.ceil(math.sqrt(total_area)),
                          max(item[2] for item in items))
        placements: Dict[str, Dict] = {}
        x = padding
        y = padding
        row_height = 0
        atlas_width = 0
        for name, tex, w, h in items:
            if x > padding and x + w > shelf_limit:
                x = padding
                y += row_height + padding
                row_height = 0
            placements[name] = {"x": x, "y": y, "width": w, "height": h}
            x += w + padding
            row_height = max(row_height, h)
            atlas_width = max(atlas_width, x)
        atlas_height = y + row_height + padding

        atlas: Texture = [[(0, 0, 0) for _ in range(atlas_width)]
                          for _ in range(atlas_height)]
        for name, tex, w, h in items:
            px = placements[name]["x"]
            py = placements[name]["y"]
            for dy in range(h):
                for dx in range(w):
                    atlas[py + dy][px + dx] = tex[dy][dx]
            placements[name]["uv"] = (
                px / atlas_width,
                py / atlas_height,
                (px + w) / atlas_width,
                (py + h) / atlas_height,
            )
        self._history.append(("texture_atlas", len(items)))
        return {
            "texture": atlas,
            "size": (atlas_width, atlas_height),
            "coordinates": placements,
        }
