def imap as f = map f as
def imap2 as bs f = map2 f as bs

type real = f64

------------------------------
---   Schedule Utilities   ---
------------------------------

def hlSched2D 't [m][n] 
  ( _lens: []i64, _strs: []i64
  , _orig: []i64, _perm: []i64
  , _sigs: []i64, _virt: []i64
  , _manifest: i64
  )
  ( xs : [m][n]t ) : [m][n]t = xs

def hlSchedFuse (_fusion_levels : []i64) x = x

def strip2 (n: i64) = (n/64, 4i64, 16i64)
def strip1 (n: i64) = (n/16, 16i64)

----------------------------------------------------------------
--- inp: [M+2][N+2]real, blur: [M+2][N]real, out: [M][N]real ---
--- blur[i, j] = inp[i, j] + inp[i, j+1] + inp[i, j+2]       ---
--- out [i, j] = blur[i,j] + blur[i+1,j] + blur[i+2,j]       ---
----------------------------------------------------------------

-- Horizontal blur (kernel 1)
def horBlur [M][N] (inp: [M+2][N+2]real) : [M+2][N]real =
  let f row j = row[j] + row[j+1] + row[j+2]
  in  imap inp (\row -> imap (iota N) (f row)) 

-- Vertical blur (kernel 2)
def verBlur [M][N] (hblur: [M+2][N]real) : [M][N]real =
  let g i j = hblur[i, j] + hblur[i+1, j] + hblur[i+2, j]
  in imap (iota M) (\i -> imap (iota N) (g i))  

def blurGold [M][N] (inp: [M+2][N+2]real) : [M][N]real =
  verBlur <| horBlur inp

-- ==
-- entry: main
-- compiled random input {1024i64 1024i64 [1026][1026]f64}
-- compiled random input {16384i64 16384i64 [16386][16386]f64}

entry main (M: i64) (N: i64) (inp: [M+2][N+2]real) : [M][N]real =
  blurGold inp
