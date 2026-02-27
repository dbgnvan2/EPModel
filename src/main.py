import numpy as np
import pygame

from engine import Simulator

# --- Rendering Constants ---
CELL_SIZE = 9          # pixels per unit cell  (100 * 9 = 900px grid)
GRID_PX   = 900        # total grid canvas width/height
SIDEBAR_X = GRID_PX    # sidebar starts here
SIDEBAR_W = 400
SCREEN_W  = GRID_PX + SIDEBAR_W
SCREEN_H  = 900


def stress_color(s_rel):
    """Map a normalised stress value [0,400] to an RGB tuple."""
    r = int(np.clip(s_rel * 1.5, 50, 255))
    g = int(np.clip(180 - s_rel * 0.8, 50, 180))
    b = int(np.clip(150 - s_rel, 50, 150))
    return (r, g, b)


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Societal Inflammatory Simulator: Multiscale Model")
    font      = pygame.font.SysFont("Courier", 14, bold=True)
    font_sm   = pygame.font.SysFont("Courier", 12)
    clock     = pygame.time.Clock()

    sim    = Simulator(10000)

    # ── Recommended defaults from Monte Carlo sweep (CDC-calibrated engine) ──
    # Best survival zone: L=1.5 + Safety Net ON + S_BASELINE=80 + E=1.0 + X=1.0
    # → 90% run survival at cycle 200 in sweep.
    sim.L                  = 1.5    # moderate-strong leadership (best: L=1.5 → 74% runs survive)
    sim.safety_net_active  = True   # +12 pts survival vs OFF; active in 75% of surviving runs
    sim.unemployment_rate  = 0.05   # 5% — realistic US baseline; 0% gives best survival
    sim.X                  = 1.0    # neutral social media (X=1.0–1.5 is optimal band)
    sim.E                  = 1.0    # neutral climate (E>1.5 drives collapse; E=1.0 → 90% survive)
    # S_BASELINE stays at class default (80) — lowest safe value; 120 halves survival rate

    log    = [
        "System Initialized  (CDC-calibrated demographics)",
        f"Age pyramid: Normal(35,15) + nursery buffer  |  Gompertz mortality",
        f"L={sim.L}  SafetyNet=ON  E={sim.E}  X={sim.X}  U={sim.unemployment_rate*100:.0f}%",
    ]
    cycle  = 0
    paused = False

    # Pre-build grid-coordinate arrays (unit index → pixel top-left)
    unit_rows = np.arange(sim.num_units) // sim.grid_size   # row in 100×100 grid
    unit_cols = np.arange(sim.num_units) %  sim.grid_size   # col in 100×100 grid
    pix_x = unit_cols * CELL_SIZE   # pixel x of top-left corner
    pix_y = unit_rows * CELL_SIZE   # pixel y of top-left corner

    while True:
        # ------------------------------------------------------------------ #
        #  Event handling                                                     #
        # ------------------------------------------------------------------ #
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:
                    sim.safety_net_active = not sim.safety_net_active
                    log.append(f"Policy: Safety Net {'ON' if sim.safety_net_active else 'OFF'}")
                if event.key == pygame.K_i:
                    sim.national_debt += 5000000
                    log.append("Action: $5M Inoculation Injection")
                if event.key == pygame.K_SPACE:
                    idx = np.random.randint(0, 10000)
                    sim.state[idx, 0] += 1000
                    log.append(f"Event: Anxiety Shock at Index {idx}")
                if event.key == pygame.K_f:
                    sim.state[:, 0] *= 1.10
                    log.append("Event: F-Key Pressed - Global Stress Spike")
                # --- Societal Factor Controls ---
                if event.key == pygame.K_l:
                    sim.L = min(sim.L + 0.1, 2.0)
                    log.append(f"Factor: Leadership (L) -> {sim.L:.1f}")
                if event.key == pygame.K_k:
                    sim.L = max(sim.L - 0.1, 0.1)
                    log.append(f"Factor: Leadership (L) -> {sim.L:.1f}")
                if event.key == pygame.K_x:
                    sim.X = min(sim.X + 0.1, 2.0)
                    log.append(f"Factor: Social Media (X) -> {sim.X:.1f}")
                if event.key == pygame.K_c:
                    sim.X = max(sim.X - 0.1, 0.1)
                    log.append(f"Factor: Social Media (X) -> {sim.X:.1f}")
                if event.key == pygame.K_e:
                    sim.E = min(sim.E + 0.1, 2.0)
                    log.append(f"Factor: Climate (E) -> {sim.E:.1f}")
                if event.key == pygame.K_d:
                    sim.E = max(sim.E - 0.1, 0.1)
                    log.append(f"Factor: Climate (E) -> {sim.E:.1f}")
                if event.key == pygame.K_p:
                    paused = not paused
                    log.append(f"{'--- PAUSED ---' if paused else '--- RESUMED ---'}")
                # --- Unemployment Rate Controls ---
                if event.key == pygame.K_u:
                    sim.unemployment_rate = min(sim.unemployment_rate + 0.01, 1.0)
                    log.append(f"Unemployment -> {sim.unemployment_rate*100:.0f}%")
                if event.key == pygame.K_j:
                    sim.unemployment_rate = max(sim.unemployment_rate - 0.01, 0.0)
                    log.append(f"Unemployment -> {sim.unemployment_rate*100:.0f}%")
                if event.key == pygame.K_g:
                    sim.coaching_active = not sim.coaching_active
                    log.append(f"Coaching: {'ON' if sim.coaching_active else 'OFF'}")

        # ------------------------------------------------------------------ #
        #  Simulation step (skipped while paused)                            #
        # ------------------------------------------------------------------ #
        if not paused:
            sim.update(cycle)
            sim.reproduce()
            cycle += 1

        # ------------------------------------------------------------------ #
        #  Build per-unit geometry masks (spec 2B + visual order of ops)     #
        # ------------------------------------------------------------------ #
        m_flat      = sim.state[:, 3]          # M account for all units
        s_flat      = sim.state[:, 0]          # Stress for all units
        s_baseline  = sim.S_BASELINE / sim.L
        departed    = sim.unit_status == sim.STATUS_DEPARTED   # ejected individuals

        # alive_mask: M > 0 (spec 2A) — used for family telemetry & square/circle logic
        alive_mask = m_flat > 0

        # Per-family live-member counts (embedded AND alive)
        embedded_alive = alive_mask & ~departed
        alive_family_counts_per_family = np.bincount(
            sim.family_ids[embedded_alive], minlength=sim.num_families
        )
        unit_family_live_count = alive_family_counts_per_family[sim.family_ids]

        # Visual Order of Operations (spec 2C):
        #   1. DEPARTED → Circle always (even if M later drains to 0)
        #   2. Embedded, alive (M>0), family >=2 → Square
        #   3. Embedded, alive (M>0), family <2  → Circle (last survivor)
        #   4. Embedded, dead (M<=0), not departed → dim red pixel (metabolic death)
        square_mask      = (~departed) & alive_mask & (unit_family_live_count >= 2)
        circle_live_mask = (~departed) & alive_mask & (unit_family_live_count <  2)
        dead_pixel_mask  = (~departed) & (~alive_mask)
        # departed circles: always drawn as circle, colour by stress if M>0 else grey
        departed_mask    = departed

        # ------------------------------------------------------------------ #
        #  Render grid canvas                                                 #
        # ------------------------------------------------------------------ #
        screen.fill((0, 0, 0))   # Black background

        radius = max(1, CELL_SIZE // 2 - 1)

        # --- 1. Departed Circles (ejected individuals — persist even after M=0) ---
        for uid in np.where(departed_mask)[0]:
            if m_flat[uid] <= 0:
                color = (0, 0, 0)      # Spec 2D Black Override: M<=0 → black, regardless of shape
            else:
                s_rel = float(np.clip(s_flat[uid] - s_baseline, 0, 400))
                color = stress_color(s_rel)
            cx = int(pix_x[uid]) + CELL_SIZE // 2
            cy = int(pix_y[uid]) + CELL_SIZE // 2
            pygame.draw.circle(screen, color, (cx, cy), radius)

        # --- 2. Squares (embedded family members) ---
        for uid in np.where(square_mask)[0]:
            if m_flat[uid] <= 0:
                color = (0, 0, 0)      # Spec 2D Black Override
            else:
                s_rel = float(np.clip(s_flat[uid] - s_baseline, 0, 400))
                color = stress_color(s_rel)
            rect = pygame.Rect(int(pix_x[uid]), int(pix_y[uid]), CELL_SIZE - 1, CELL_SIZE - 1)
            pygame.draw.rect(screen, color, rect)

        # --- 3. Live singleton circles (last survivors still embedded) ---
        for uid in np.where(circle_live_mask)[0]:
            if m_flat[uid] <= 0:
                color = (0, 0, 0)      # Spec 2D Black Override
            else:
                s_rel = float(np.clip(s_flat[uid] - s_baseline, 0, 400))
                color = stress_color(s_rel)
            cx = int(pix_x[uid]) + CELL_SIZE // 2
            cy = int(pix_y[uid]) + CELL_SIZE // 2
            pygame.draw.circle(screen, color, (cx, cy), radius)

        # --- 4. Metabolic deaths (M<=0, not departed) — 1px dim red marker ---
        for uid in np.where(dead_pixel_mask)[0]:
            screen.set_at((int(pix_x[uid]), int(pix_y[uid])), (30, 0, 0))

        # ------------------------------------------------------------------ #
        #  Sidebar Dashboard                                                  #
        # ------------------------------------------------------------------ #
        pygame.draw.rect(screen, (10, 10, 20), (SIDEBAR_X, 0, SIDEBAR_W, SCREEN_H))

        alive_count   = int(np.sum(alive_mask))
        dead_count    = sim.total_deaths
        avg_s         = float(np.mean(s_flat[alive_mask])) if alive_count > 0 else 0.0
        circle_count  = int(np.sum(departed & alive_mask))
        recovery_act  = int(np.sum(sim.recovery_timers > 0))
        survival_pct  = alive_count / sim.num_units * 100.0
        net_pop       = sim.total_births - sim.total_deaths   # + growing, - shrinking

        stats = [
            "=== LIVE STATUS ===",
            f"CYCLE       : {cycle}  {'[PAUSED]' if paused else ''}",
            f"LIVE UNITS  : {alive_count:,}  ({survival_pct:.1f}%)",
            f"NET POP     : {net_pop:+,}  (B-D)",
            f"TOTAL DEATHS: {dead_count:,}",
            f"AVG STRESS  : {avg_s:.1f}",
            f"CIRCLES OUT : {circle_count:,}",
            "",
            "=== DEMOGRAPHICS ===",
            f"AVG AGE     : {sim.avg_age:.1f} yrs",
            f"BIRTHS      : {sim.total_births:,}",
            f"MARRIAGES   : {sim.total_marriages:,}",
            f"DIVORCES    : {sim.divorce_count:,}",
            f"UNEMPLOYMENT: {sim.unemployment_rate*100:.0f}%  [j/u]",
            f"HOMELESS    : {sim.homelessness_rate:.1f}%",
            "",
            "=== FAMILY ===",
            f"ACTIVE FAMS : {sim.active_family_count:,}",
            f"AVG FAM SIZE: {sim.avg_family_size:.2f}",
            f"RECOVERY ACT: {recovery_act:,}",
            "",
            "=== ECONOMY ===",
            f"AVG M       : {sim.avg_family_m:,.0f}",
            f"DEBT        : ${int(sim.national_debt):,}",
            f"SAFETY NET  : {'ACTIVE [n]' if sim.safety_net_active else 'OFF    [n]'}",
            f"COACHING    : {'ACTIVE [g]' if sim.coaching_active else 'OFF    [g]'}",
            f"C GROWTH    : {'+0.5/yr' if sim.coaching_active else '+0.0/yr'}",
            "",
            "=== FACTORS ===",
            f"L (Lead) : {sim.L:.1f}  [k/l]",
            f"X (Media): {sim.X:.1f}  [c/x]",
            f"E (Clim) : {sim.E:.1f}  [d/e]",
            f"AVG C    : {sim.avg_c:.1f}",
            "",
            "=== SHAPE KEY ===",
            "  SQUARE = Family Member",
            "  CIRCLE = Separated",
            "  BLACK  = Dead / M<=0",
        ]

        y_cursor = 10
        for txt in stats:
            if txt.startswith("==="):
                color = (255, 220, 50)
            elif txt == "":
                y_cursor += 6
                continue
            elif "DEBT" in txt and sim.national_debt > 0:
                color = (255, 80, 80)
            elif "NET POP" in txt:
                color = (100, 255, 100) if net_pop >= 0 else (255, 100, 100)
            elif "SURVIVAL" in txt or ("%" in txt and "LIVE UNITS" in txt):
                color = (100, 200, 255)
            else:
                color = (220, 220, 220)
            screen.blit(font.render(txt, True, color), (SIDEBAR_X + 10, y_cursor))
            y_cursor += 22

        # Divider
        pygame.draw.line(screen, (60, 60, 80), (SIDEBAR_X + 5, y_cursor), (SCREEN_W - 5, y_cursor))
        y_cursor += 8

        # Scrolling event log
        screen.blit(font.render("=== EVENT LOG ===", True, (255, 220, 50)), (SIDEBAR_X + 10, y_cursor))
        y_cursor += 22
        visible_log = log[-20:]
        for entry in visible_log:
            screen.blit(font_sm.render(entry[:48], True, (130, 255, 130)), (SIDEBAR_X + 10, y_cursor))
            y_cursor += 16

        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()
