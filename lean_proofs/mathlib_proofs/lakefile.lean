import Lake
open Lake DSL

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.29.0"

package «formInvMathlib» where

lean_lib «FormInvMathlib» where
  roots := #[`FormInvMathlib]
