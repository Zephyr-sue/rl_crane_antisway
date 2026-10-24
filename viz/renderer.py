"""pygame renderer for the crane: used by env.render() and the interactive live demo."""
import os

import numpy as np

WIDTH, HEIGHT = 960, 540
SCALE = 70.0           # px per metre
RAIL_Y = 110
BG = (250, 250, 248)
INK = (11, 11, 11)
INK_2 = (82, 81, 78)
GRID = (228, 227, 223)
TROLLEY = (42, 120, 214)
HOOK = (82, 81, 78)
PAYLOAD = (235, 104, 52)
TARGET = (27, 175, 122)
GUST = (98, 80, 214)


class CraneRenderer:
    def __init__(self, headless: bool = False, title: str = "Double-pendulum crane"):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        import pygame
        self.pg = pygame
        pygame.init()
        self.headless = headless
        if headless:
            self.screen = pygame.Surface((WIDTH, HEIGHT))
        else:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 16)
        self.big = pygame.font.SysFont("consolas,menlo,monospace", 20, bold=True)
        self.clock = pygame.time.Clock()

    def to_px(self, x, y):
        return int(WIDTH / 2 + x * SCALE), int(RAIL_Y + y * SCALE)

    def draw(self, state, dynamics, target_x, u, gust, t, hud_lines=(), title=None, fps=50):
        pg, s = self.pg, self.screen
        x, _, th1, _, th2, _ = state
        L1, L2 = dynamics.L1, dynamics.L2
        s.fill(BG)

        # Rail with metre ticks and end-stops
        pg.draw.line(s, INK_2, self.to_px(-6.2, 0), self.to_px(6.2, 0), 4)
        for m in range(-6, 7):
            px = self.to_px(m, 0)
            pg.draw.line(s, GRID, (px[0], px[1] + 6), (px[0], HEIGHT - 40), 1)
            s.blit(self.font.render(f"{m}", True, INK_2), (px[0] - 6, HEIGHT - 32))
        for end in (-6.0, 6.0):
            px = self.to_px(end, 0)
            pg.draw.rect(s, INK, (px[0] - 4, px[1] - 14, 8, 28))

        # Target marker
        tx = self.to_px(target_x, 0)[0]
        pg.draw.line(s, TARGET, (tx, RAIL_Y + 10), (tx, HEIGHT - 40), 3)
        s.blit(self.font.render("target", True, TARGET), (tx + 6, HEIGHT - 60))

        # Kinematics
        hook = (x + L1 * np.sin(th1), L1 * np.cos(th1))
        load = (hook[0] + L2 * np.sin(th2), hook[1] + L2 * np.cos(th2))
        p_trolley, p_hook, p_load = self.to_px(x, 0), self.to_px(*hook), self.to_px(*load)

        pg.draw.line(s, INK_2, p_trolley, p_hook, 2)
        pg.draw.line(s, INK_2, p_hook, p_load, 2)
        pg.draw.rect(s, TROLLEY, (p_trolley[0] - 30, p_trolley[1] - 16, 60, 26), border_radius=4)
        pg.draw.circle(s, HOOK, p_hook, 8)
        size = int(14 + 6 * np.sqrt(dynamics.m2))
        pg.draw.rect(s, PAYLOAD, (p_load[0] - size // 2, p_load[1] - size // 2, size, size), border_radius=4)

        # Force arrow on trolley, gust arrow on payload
        self._arrow((p_trolley[0], p_trolley[1] - 30), u * 90, TROLLEY)
        if abs(gust) > 1e-6:
            self._arrow((p_load[0] - int(np.sign(gust) * 70), p_load[1]), gust * 12, GUST, width=4)
            s.blit(self.font.render(f"gust {gust:+.1f} N", True, GUST), (p_load[0] - 50, p_load[1] + size))

        # HUD
        lines = [
            f"t = {t:5.2f} s   F = {u * 30:+6.1f} N",
            f"hook {np.rad2deg(th1):+6.1f} deg   payload {np.rad2deg(th2):+6.1f} deg",
            f"m2 = {dynamics.m2:.2f} kg   L1 = {L1:.2f} m   L2 = {L2:.2f} m",
            *hud_lines,
        ]
        if title:
            s.blit(self.big.render(title, True, INK), (16, 12))
        for i, line in enumerate(lines):
            s.blit(self.font.render(line, True, INK), (WIDTH - 470, 12 + 20 * i))

        if self.headless:
            return np.transpose(pg.surfarray.array3d(s), (1, 0, 2))
        pg.event.pump()
        pg.display.flip()
        self.clock.tick(fps)
        return None

    def _arrow(self, start, length, color, width=3):
        pg = self.pg
        if abs(length) < 2:
            return
        end = (int(start[0] + length), start[1])
        pg.draw.line(self.screen, color, start, end, width)
        d = 1 if length > 0 else -1
        pg.draw.polygon(self.screen, color, [end, (end[0] - d * 10, end[1] - 6), (end[0] - d * 10, end[1] + 6)])

    def close(self):
        self.pg.quit()
