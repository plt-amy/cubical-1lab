<!--
```agda
open import Cat.Prelude

open import Order.Diagram.Glb
open import Order.Diagram.Lub
open import Order.Base

import Order.Reasoning as Pos
```
-->

```agda
module Order.Lattice.Complete {o ℓ} (P : Poset o ℓ) where
```

<!--
```agda
open Pos P
open is-lub
open is-glb
open Lub
open Glb
```
-->

# Complete lattices {defines="complete-lattice"}

Unlike with the tangible difference [[join]] and [[meet]] semilattices,
even as objects, a partially ordered set is an inf-lattice precisely if
it is a [[suplattice]]. This module is dedicated to recording that fact,
but the proof is almost elementary.

We will elaborate on the proof starting from the existence of [[greatest
lower bounds]]. We have a family $F : I \to P$ into some inf-lattice
$P$, and we want to compute its [[least upper bound]]. But this is
precisely a _lower bound_ on the set of _upper bounds_ for $F$:

```agda
is-complete→is-cocomplete
  : (glbs : ∀ {I : Type o} (F : I → Ob) → Glb P F)
  → ∀ {I : Type o} (F : I → Ob) → Lub P F
is-complete→is-cocomplete glbs {I} F = join where
  I' : Type o
  I' = Σ[ o ∈ Ob ] □ ((i : I) → F i ≤ o)

  meet : Glb P fst
  meet = glbs {I'} fst
```

Despite the nominal coincidence, we still have to show the universal
property: call the meet we constructed above $m$. Suppose we have $i :
I$. We can show that $F(i) \le m$ by the universal property for showing
things are *below* $m$: this reduces to showing that, given some $j : P$
satisfying $\forall i', F i \le j$, we have $F i \le j$.

```agda
  join : Lub P F
  join .lub = meet .glb
  join .has-lub .fam≤lub   i = meet .greatest (F i) λ (j , p) → out! p i
```

We must now show that, if we have $u$ with $\forall i', F i \le u$, $m
\le u$. But this is, again, immediate:

```agda
  join .has-lub .least ub' x = meet .glb≤fam (ub' , inc x)
```

The proof of the converse direction, starting from a suplattice, is
formally dual.

```agda
is-cocomplete→is-complete
  : (glbs : ∀ {I : Type o} (F : I → Ob) → Lub P F)
  → ∀ {I : Type o} (F : I → Ob) → Glb P F
is-cocomplete→is-complete lubs {I} F = meet where
  I' : Type o
  I' = Σ[ o ∈ Ob ] □ ((i : I) → o ≤ F i)

  join : Lub P fst
  join = lubs {I'} fst

  meet : Glb P F
  meet .glb = join .lub
  meet .has-glb .glb≤fam i = join .least (F i) (λ (i , p) → out! p _)
  meet .has-glb .greatest ub' x = join .fam≤lub (ub' , inc x)
```
