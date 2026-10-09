import "blur-gold"

-----------------
--- Schedules
-----------------

-- schedule for horizontal blur (kernel 1)
def horSchedule (M: i64) (N: i64) = -- : ([M+2][N]real -> [M+2][N]real) =
  let dims = [M, 3, N] -- LMAD's dimensions
  let strd = [N, N, 1] -- LMAD's total strides
  let orig = [0, 0, 1] -- correspondence with original dimensions
  let perm = [0, 2, 1] -- permutation (scatter semantics)
  let sigs = [0, 0, 0] -- signals
  -- ^ results in LMAD = [(M:N), (N,1), (3:N)]
  in  hlSched2D (dims, orig, perm, sigs, strd, [], 0)
  -- ^ the last 0 means substitute the new shape fwd

-----------------------------------
--- Main Kernel and entrypoints ---
-----------------------------------

-- Currently the schedules and fusion are not supported
--   uncomment when this changes
def blurFullyFused [M][N] (inp: [M+2][N+2]real) : [M][N]real =
  horBlur inp
  |> horSchedule M N
  |> hlSchedFuse [1, 1]
  |> verBlur

-- ==
-- entry: main
-- compiled random input {1024i64 1024i64 [1026][1026]f64}
-- compiled random input {16384i64 16384i64 [16386][16386]f64}

-- auto output

entry main (M: i64) (N: i64) (inp: [M+2][N+2]real) : [M][N]real =
  blurFullyFused inp

-- ==
-- entry: valid
-- compiled random input {1024i64 1024i64 [1026][1026]f64}
-- output{ true }
-- compiled random input {16384i64 16384i64 [16386][16386]f64}
-- output{ true }

-- entry valid (M: i64) (N: i64) (input: [M+2][N+2]real) : bool =
--   let fully_fused = opaque <| blurFullyFused input
--   let orig_unfused= opaque <| blurGold input
--   in  fully_fused == orig_unfused
