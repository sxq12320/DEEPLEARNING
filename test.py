import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

fig, ax = plt.subplots(figsize=(7.2, 10.2))
ax.set_xlim(0, 10); ax.set_ylim(0, 15); ax.axis('off')

# 颜色：原图风格 + 改进高亮
C_BG   = '#e8f3e3'   # 外框浅绿
C_OLD  = '#dfe3f2'   # 原有算子：淡蓝紫
C_NEW  = '#ffd9b3'   # 改进算子：橙（高亮）
C_OP   = '#ffffff'   # 圆形算子底
EDGE   = '#333333'

# 外框（虚线圆角）
ax.add_patch(FancyBboxPatch((0.4, 0.4), 9.2, 14.2, boxstyle="round,pad=0.02,rounding_size=0.4",
             fc=C_BG, ec='#7fae6f', ls='--', lw=1.6, zorder=0))

def box(x, y, w, h, text, new=False, fs=10):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                 fc=(C_NEW if new else C_OLD), ec=EDGE, lw=1.2, zorder=2))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, zorder=3,
            fontweight=('bold' if new else 'normal'))

def op(x, y, sym, r=0.32):
    ax.add_patch(Circle((x, y), r, fc=C_OP, ec=EDGE, lw=1.2, zorder=2))
    ax.text(x, y, sym, ha='center', va='center', fontsize=12, zorder=3)

def arr(p1, p2, text='', tpos=0.5, dx=0.0, dy=0.0, col=EDGE, style='-|>', lw=1.3):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14, color=col, lw=lw, zorder=1)
    ax.add_patch(a)
    if text:
        mx = p1[0]+(p2[0]-p1[0])*tpos+dx; my = p1[1]+(p2[1]-p1[1])*tpos+dy
        ax.text(mx, my, text, ha='center', va='center', fontsize=8.5, color='#444', zorder=4)

def txt(x, y, s, fs=9, col='#222', ha='center'):
    ax.text(x, y, s, ha=ha, va='center', fontsize=fs, color=col, zorder=4)

# ---- 节点（y 从上到下 = 数据流方向）----
txt(5, 14.3, r'input  $c_{in}$', fs=11)
op(5, 13.3, 'Split'); txt(7.0, 13.55, r'$\tau_p$ / $1-\tau_p$', fs=8.5, ha='left')

# 左路 τp
box(2.3, 8.6, 2.0, 0.8, 'Conv3x3 (DW)')
# 右路统计：改进一（橙色）
box(5.0, 11.3, 1.5, 0.7, 'Std', new=False)          # 原 std 基因保留
box(7.0, 11.3, 1.5, 0.7, 'Mean', new=False)
box(5.0, 9.9, 3.4, 0.8, 'local multi-scale\n+ multi-freq DCT', new=True)  # 改进一
op(6.0, 10.6, 'U')
box(6.0, 8.9, 2.2, 0.7, r'Conv1x(2+k)', new=True)    # 改进一：2 -> 2+k

# 权重生成：改进二（橙色）
box(6.0, 7.5, 2.4, 0.7, 'H-Sigmoid -> (c,H,W)', new=True)  # 改进二
op(6.0, 6.3, '⊙')
box(6.0, 5.3, 1.4, 0.7, 'BN')

# 左路门控：改进三（橙色）
op(2.3, 6.3, '⊙'); txt(2.3, 6.95, r'$1+\alpha w$', fs=8)

# 末端 concat
op(4.2, 3.6, 'U')
txt(4.2, 2.7, r'output  $c_{in}$', fs=11)

# ---- 连线 ----
arr((5,14.0),(5,13.62))
arr((4.4,13.0),(2.3,9.05), r'$\tau_p$', tpos=0.25, dx=-0.5)         # split -> 左 Conv3x3
arr((5.6,13.0),(6.0,11.65), r'$1-\tau_p$', tpos=0.2, dx=0.7)        # split -> 右统计
arr((5.0,10.95),(5.4,10.78)); arr((7.0,10.95),(6.6,10.78))          # Std/Mean -> U
arr((6.0,10.28),(6.0,10.30)); arr((6.0,10.28),(6.0,9.25)+ (0,0))    # U -> 改进一框(占位)
# 上面两行用更直接的连法覆盖：
arr((6.0,11.0),(6.0,10.32))                                         # 统计 -> U
arr((6.0,10.28),(6.0,9.30))                                         # U -> 改进一
arr((5.0,9.5),(5.4,9.30)); arr((7.0,9.5),(6.6,9.30))                # 局部/频率汇入改进一(示意)
arr((6.0,8.55),(6.0,7.85))                                          # Conv1x(2+k) -> H-Sig
arr((6.0,7.15),(6.0,6.62))                                          # H-Sig -> ⊙
arr((8.6,13.0),(8.6,6.3)); arr((8.6,6.3),(6.32,6.3))                # 右路原始特征旁路 -> ⊙
arr((6.0,5.98),(6.0,5.65))                                          # ⊙ -> BN
arr((6.0,4.95),(4.5,3.92))                                          # BN -> 末端U
arr((2.3,8.2),(2.3,6.62))                                           # 左Conv3x3 -> 左⊙(门控)
arr((6.0,7.5),(2.62,6.3), col='#c0651f', style='-|>')               # 改进三：权重共享给左路
arr((2.3,5.98),(2.3,3.92)); arr((2.3,3.7),(3.88,3.6))               # 左门控 -> 末端U
arr((4.2,3.28),(4.2,3.0))                                           # 末端U -> output

# ---- 图例 ----
lx, ly = 0.7, 1.6
ax.add_patch(FancyBboxPatch((lx, ly), 0.5, 0.4, boxstyle="round,pad=0.02", fc=C_OLD, ec=EDGE)); txt(lx+0.9, ly+0.2, 'original operator', fs=8.5, ha='left')
ax.add_patch(FancyBboxPatch((lx, ly-0.6), 0.5, 0.4, boxstyle="round,pad=0.02", fc=C_NEW, ec=EDGE)); txt(lx+0.9, ly-0.4, 'our improvement (I/II/III)', fs=8.5, ha='left')
txt(lx+0.0, ly-1.25, 'U = concat    ⊙ = element-wise mul    ⊕ = add', fs=8.5, ha='left')

# 改进编号标注
txt(8.9, 9.9, 'I',  fs=11, col='#c0392b'); 
txt(8.6, 7.5, 'II', fs=11, col='#c0392b');
txt(1.3, 6.3, 'III',fs=11, col='#c0392b');

plt.tight_layout(); plt.savefig('PAT_ch_prime.png', dpi=200, bbox_inches='tight'); plt.show()