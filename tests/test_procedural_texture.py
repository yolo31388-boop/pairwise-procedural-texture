"""
游戏内程序化纹理生成与材质系统 - 验收测试
运行方式：python -m pytest tests/test_procedural_texture.py -q
共 14 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rendering.texture import ProceduralTexture


def make_texture(colors):
    """用颜色矩阵构造纹理。"""
    return [[colors[y][x] for x in range(len(colors[0]))] for y in range(len(colors))]


class TestProceduralTexture:
    def setup_method(self):
        self.system = ProceduralTexture()

    # ---- 噪声函数：Perlin + 分形 ----
    def test_case_01(self):
        # Perlin 噪声确定性且值域为 [-1, 1]
        sys_ = self.system
        for i in range(10):
            x, y = i * 0.37, i * 0.91
            v1 = sys_.noise_function(x, y, noise_type="perlin", seed=42)
            v2 = sys_.noise_function(x, y, noise_type="perlin", seed=42)
            assert v1 == v2
            assert -1.0 <= v1 <= 1.0

    def test_case_02(self):
        # Perlin 噪声平滑连续，不是简单随机数；不同种子结果不同
        sys_ = self.system
        prev = sys_.noise_function(0.0, 0.0, noise_type="perlin", seed=1)
        for i in range(1, 50):
            cur = sys_.noise_function(i * 0.01, 0.0, noise_type="perlin", seed=1)
            assert abs(cur - prev) < 0.2  # 平滑：相邻采样不会剧烈跳变
            prev = cur
        a = sys_.noise_function(1.3, 2.7, noise_type="perlin", seed=1)
        b = sys_.noise_function(1.3, 2.7, noise_type="perlin", seed=2)
        assert a != b

    def test_case_03(self):
        # 分形噪声：多倍频叠加，细节比单层丰富，值域仍为 [-1, 1]
        sys_ = self.system
        diffs = 0
        for i in range(20):
            x, y = i * 0.53, i * 0.29
            f1 = sys_.noise_function(x, y, noise_type="fractal", seed=7, octaves=1)
            f4 = sys_.noise_function(x, y, noise_type="fractal", seed=7, octaves=4)
            assert -1.0 <= f1 <= 1.0
            assert -1.0 <= f4 <= 1.0
            if f1 != f4:
                diffs += 1
        assert diffs > 10  # 分形细节改变了绝大多数采样点

    # ---- 纹理生成：算法 + 参数 ----
    def test_case_04(self):
        # 每种算法都能生成尺寸正确、像素合法的纹理
        for algo in ("noise", "checker", "brick", "wood", "marble"):
            tex = self.system.generate_texture(16, 12, algorithm=algo)
            assert len(tex) == 12
            assert all(len(row) == 16 for row in tex)
            for row in tex:
                for pixel in row:
                    assert len(pixel) == 3
                    assert all(isinstance(c, int) and 0 <= c <= 255 for c in pixel)

    def test_case_05(self):
        # 参数可调：不同种子/缩放生成不同纹理
        t1 = self.system.generate_texture(16, 16, "noise", {"seed": 1})
        t2 = self.system.generate_texture(16, 16, "noise", {"seed": 2})
        t3 = self.system.generate_texture(16, 16, "noise", {"seed": 1, "scale": 16.0})
        assert t1 != t2
        assert t1 != t3
        # 相同参数可复现
        assert t1 == self.system.generate_texture(16, 16, "noise", {"seed": 1})

    def test_case_06(self):
        # 棋盘格算法按参数精确生成
        tex = self.system.generate_texture(
            8, 8, "checker",
            {"cell_size": 2, "color_a": (255, 0, 0), "color_b": (0, 0, 255)})
        assert tex[0][0] == (255, 0, 0)
        assert tex[0][1] == (255, 0, 0)
        assert tex[0][2] == (0, 0, 255)
        assert tex[2][0] == (0, 0, 255)
        assert tex[3][3] == (255, 0, 0)

    # ---- 材质合成：层 + 属性 ----
    def test_case_07(self):
        # 多层法线混合：红色纹理 + 50% 蓝色层
        red = make_texture([[(255, 0, 0), (255, 0, 0)], [(255, 0, 0), (255, 0, 0)]])
        mat = self.system.material_synthesis([
            {"texture": red},
            {"color": (0, 0, 255), "opacity": 0.5},
        ])
        assert mat["layers"] == 2
        assert mat["size"] == (2, 2)
        assert mat["color"][0][0] == (128, 0, 128)

    def test_case_08(self):
        # 材质属性：粗糙度、金属度、法线；multiply 混合模式
        white = make_texture([[(255, 255, 255)]])
        normal_map = make_texture([[(128, 128, 255)]])
        mat = self.system.material_synthesis(
            [
                {"texture": white},
                {"color": (255, 0, 0), "blend_mode": "multiply"},
            ],
            {"roughness": 0.8, "metallic": 0.3, "normal": normal_map},
        )
        assert mat["color"][0][0] == (255, 0, 0)  # 白 * 红 = 红
        assert mat["roughness"] == pytest.approx(0.8)
        assert mat["metallic"] == pytest.approx(0.3)
        assert mat["normal"] == normal_map

    def test_case_09(self):
        # 层叠顺序影响结果；add 混合模式
        red = make_texture([[(255, 0, 0)]])
        blue = make_texture([[(0, 0, 255)]])
        m1 = self.system.material_synthesis([
            {"texture": red},
            {"texture": blue, "opacity": 0.75},
        ])
        m2 = self.system.material_synthesis([
            {"texture": blue},
            {"texture": red, "opacity": 0.75},
        ])
        assert m1["color"][0][0] == (64, 0, 191)
        assert m2["color"][0][0] == (191, 0, 64)
        m3 = self.system.material_synthesis([
            {"texture": make_texture([[(100, 100, 100)]])},
            {"color": (100, 100, 100), "blend_mode": "add"},
        ])
        assert m3["color"][0][0] == (200, 200, 200)

    # ---- 纹理映射：UV + 过滤 ----
    def test_case_10(self):
        # UV 坐标精确映射到纹素（最近邻）
        tex = make_texture([
            [(255, 0, 0), (0, 255, 0)],
            [(0, 0, 255), (255, 255, 255)],
        ])
        assert self.system.texture_mapping(tex, 0.25, 0.25, "nearest") == (255, 0, 0)
        assert self.system.texture_mapping(tex, 0.75, 0.25, "nearest") == (0, 255, 0)
        assert self.system.texture_mapping(tex, 0.25, 0.75, "nearest") == (0, 0, 255)
        assert self.system.texture_mapping(tex, 0.75, 0.75, "nearest") == (255, 255, 255)

    def test_case_11(self):
        # 双线性过滤：纹素之间平滑插值，放大不再是马赛克
        tex = make_texture([
            [(0, 0, 0), (200, 200, 200)],
            [(0, 0, 0), (200, 200, 200)],
        ])
        mid = self.system.texture_mapping(tex, 0.5, 0.5, "bilinear")
        assert mid == pytest.approx((100.0, 100.0, 100.0))
        quarter = self.system.texture_mapping(tex, 0.375, 0.5, "bilinear")
        assert quarter == pytest.approx((50.0, 50.0, 50.0))
        # 插值结果介于两个纹素之间（非马赛克跳变）
        assert 0 < quarter[0] < mid[0] < 200

    def test_case_12(self):
        # 环绕模式：repeat 让 UV 超出 [0,1] 时平铺；clamp 截断
        tex = make_texture([
            [(255, 0, 0), (0, 255, 0)],
            [(0, 0, 255), (255, 255, 255)],
        ])
        assert self.system.texture_mapping(tex, 1.25, 0.25, "nearest", "repeat") == (255, 0, 0)
        assert self.system.texture_mapping(tex, -0.25, 0.25, "nearest", "repeat") == (0, 255, 0)
        assert self.system.texture_mapping(tex, 5.0, 0.25, "nearest", "clamp") == (0, 255, 0)
        with pytest.raises(ValueError):
            self.system.texture_mapping(tex, 0.5, 0.5, "cubic")

    # ---- 纹理图集：打包 + 坐标 ----
    def test_case_13(self):
        # 多个纹理打包进一张图集，坐标不重叠且尺寸正确
        textures = {
            "a": make_texture([[(255, 0, 0)] * 4] * 4),
            "b": make_texture([[(0, 255, 0)] * 2] * 2),
            "c": make_texture([[(0, 0, 255)] * 3] * 1),
        }
        atlas = self.system.texture_atlas(textures)
        aw, ah = atlas["size"]
        assert len(atlas["texture"]) == ah
        assert all(len(row) == aw for row in atlas["texture"])
        coords = atlas["coordinates"]
        assert set(coords) == {"a", "b", "c"}
        boxes = []
        for name, tex in textures.items():
            c = coords[name]
            assert c["width"] == len(tex[0]) and c["height"] == len(tex)
            boxes.append((c["x"], c["y"], c["x"] + c["width"], c["y"] + c["height"]))
            assert 0 <= c["x"] and 0 <= c["y"]
            assert c["x"] + c["width"] <= aw and c["y"] + c["height"] <= ah
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]

    def test_case_14(self):
        # 图集坐标可还原原始纹理像素，UV 在 [0,1] 内
        textures = {
            "red": make_texture([[(255, 0, 0), (200, 0, 0)]]),
            "green": make_texture([[(0, 255, 0)], [(0, 200, 0)]]),
        }
        atlas = self.system.texture_atlas(textures)
        grid = atlas["texture"]
        aw, ah = atlas["size"]
        for name, tex in textures.items():
            c = atlas["coordinates"][name]
            for dy in range(c["height"]):
                for dx in range(c["width"]):
                    assert grid[c["y"] + dy][c["x"] + dx] == tex[dy][dx]
            u0, v0, u1, v1 = c["uv"]
            assert 0.0 <= u0 < u1 <= 1.0
            assert 0.0 <= v0 < v1 <= 1.0
            assert u0 == pytest.approx(c["x"] / aw)
            assert v1 == pytest.approx((c["y"] + c["height"]) / ah)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
