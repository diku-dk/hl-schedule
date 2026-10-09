import "blur-gold"

-----------------
--- Schedules
-----------------

-- Schedule for horizontal blur (kernel 1)
def horSchedule (M: i64) (m: i64) (Tm: i64) (N: i64) (n: i64) (Tn: i64) : ([M+2][N]real -> [M+2][N]real) =
  let dims = [   m,    Tm+2, n, Tn] -- LMAD's dimensions
  let strd = [Tm*n*Tn, n*Tn, Tn, 1] -- LMAD's total strides
  let orig = [   0,     0,   1,  1] -- correspondence with original dimensions
  let perm = [   0,     2,   1,  3] -- permutation (scatter semantics)
  let sigs = [   0,     0,   0,  0] -- signals
  -- ^ results in LMAD = [(m : Tm*n*Tn), (n: Tn), (Tm+2: n*Tn), (Tn, 1)]
  in  hlSched2D (dims, orig, perm, sigs, strd, [], 0) -- substitute new shape fwd

-- Schedule for vertical blur (kernel 2)
def verSchedule (M: i64) (m: i64) (Tm: i64) (N: i64) (n: i64) (Tn: i64) : ([M][N]real -> [M][N]real) =
  let dims = [   m,     Tm,  n, Tn] -- LMAD's dimensions
  let strd = [Tm*n*Tn, n*Tn, Tn, 1] -- LMAD's total strides
  let orig = [   0,     0,   1,  1] -- correspondence with original dimensions
  let perm = [   0,     2,   1,  3] -- permutation (scatter semantics)
  let sigs = [   0,     0,   0,  0] -- signals
  -- ^ results in LMAD = [(m : Tm*n*Tn), (n: Tn), (Tm: n*Tn), (Tn, 1)]
  in  hlSched2D (dims, orig, perm, sigs, strd, [], 1) -- manifest in the original shape

-----------------------------------
--- Main Kernel and entrypoints ---
-----------------------------------

-- Currently the schedules and fusion are not supported
--   uncomment when this changes
def blurOverlapTile [M][N] (inp: [M+2][N+2]real) : [M][N]real =
  let (m, Tm) = strip1 M
  let (n, Tn) = strip1 N
  let h_blur = horBlur inp
--            |> horSchedule M m Tm N n Tn
--            |> hlSchedFuse [1, 1]
            -- ^ fuse the first two levels classicaly
  let v_blur = verBlur h_blur
--            |> verSchedule M m Tm N n Tn
  in  v_blur

-- ==
-- entry: main
-- compiled random input {1024i64 1024i64 [1026][1026]f64}
-- compiled random input {16384i64 16384i64 [16386][16386]f64}

-- auto output

entry main (M: i64) (N: i64) (inp: [M+2][N+2]real) : [M][N]real =
  blurOverlapTile inp

-- ==
-- entry: valid
-- compiled random input {1024i64 1024i64 [1026][1026]f64}
-- output{ true }
-- compiled random input {16384i64 16384i64 [16386][16386]f64}
-- output{ true }

entry valid (M: i64) (N: i64) (input: [M+2][N+2]real) : bool =
  let fully_fused = opaque <| blurOverlapTile input
  let orig_unfused= opaque <| blurGold input
  in  fully_fused == orig_unfused

-- futhark bench --backend=cuda --pass-option=--device=#1 blur-overlap-tiling.fut
