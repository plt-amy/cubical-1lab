<!--
```agda
open import 1Lab.Prelude

open import Order.Diagram.Glb
open import Order.Diagram.Lub
open import Order.Suplattice
open import Order.Subposet
open import Order.Base hiding (pres-≤)

import Order.Reasoning as Pos
```
-->

```agda
module Order.Diagram.Closure where
```

# Closure operators

One of the most practical ways to obtain a subset which should be
*closed under* some operations, e.g. a [[subgroup]], is to start with
some subset $s \sube G$, and inductively add the missing elements --- in
this case, the products $ij$ for $i \in s$, $j \in s$, and the inverses
$i^{-1}$. This *subgroup completion*, which we will momentarily write
$c(-)$, has a few order-theoretic properties:

- The original subset $s$ is included in $c(s)$. Intuitively this is
because $c(s)$ only *adds* the missing elements, and does not take
anything away from the set.

- The subgroup completion also respects inclusion: if we have $s \sube
t$, then we will have $c(s) \sube c(t)$.

- Finally, the subgroup completion is *idempotent*: if we start with an
arbitrary $s \sube G$, complete it to a sub*group* $c(s)$, then any
further applications $cc(s)$ will have no effect --- $c(s)$ is already a
subgroup, so it's not missing anything!

::: {.definition #closure-operator}
The notion of **closure operator** on a [[partially ordered set]] $P$
generalises these conditions away from subsets and inclusion. A closure
operator $j$ is a [[monotone map]] $j : P \to P$ which additionally
satisfies **extensivity** ($x \le jx$) and **idempotence** ($jjx \le
jx$).
:::

The categorically-inclined reader will have noticed that a closure
operator on $P$ is precisely a [[monad]] on $P$-qua-category. As a nod,
we baptise the formalised counterparts of extensivity and idempotence
after the unit and multiplication in a monad: they are $\eta$ and $\mu$.

```agda
record Closure {o ℓ} (P : Poset o ℓ) : Type (o ⊔ ℓ) where
  open Poset P
  field
    close  : Ob → Ob
    pres-≤ : ∀ {x y} → x ≤ y → close x ≤ close y
    η : ∀ {x} → x ≤ close x
    μ : ∀ {x} → close (close x) ≤ close x
```

In a monad on an arbitrary category, we have maps $jx \to jjx$ and $jjx
\to jx$; but these need not be inverses in general.^[As an example, take
the list monad, where we have `map pure (concat [[1,2,3]]) =
[[1],[2],[3]]`{.Haskell}] However, in a partially-ordered set, this
possibility is quashed, and our weakening of the idempotence condition
to an inequality was inessential:

```agda
  close-idempotent : ∀ {x} → close x ≡ close (close x)
  close-idempotent = ≤-antisym (pres-≤ η) μ
```

## Closed elements {defines="closed-element fixset"}

A ($j$-)**closed element** $x : P$ is one satisfying $x = jx$. Since we
have $x \le j x$, the $j$-closure condition is equivalent to demanding
$jx \le x$. Therefore, a closed element is the posetal analogue of a
[[algebra over the monad|monad algebra]] $j$.

```agda
  is-closed : Ob → Prop o
  is-closed x .∣_∣   = x ≡ close x
  is-closed x .is-tr = hlevel!
```

Accordingly, we also have the analogue of the [[Eilenberg-Moore
category]] of $j$: its **fixset**. The need to distinguish between the
underlying type and the actual order compels us to name the latter the
**fixposet** of $j$, and write it $P^j$.

```agda
  Fixset : Type o
  Fixset = Σ[ o ∈ Ob ] (o ≡ close o)

  Fixposet : Poset o ℓ
  Fixposet = Subposet P is-closed

  unit : Ob → Fixset
  unit u .fst = close u
  unit u .snd = close-idempotent
```

<!--
```agda
module _ {o ℓ} {P : Poset o ℓ} (j : Closure P) where
  private module j = Closure j
  open is-lub
  open is-glb
  open Pos P
```
-->

## Meets and joins

If the underlying poset $P$ has [[meets]] --- or [[joins]] --- then the
fixposet $P^j$ does as well. While the meet $x\land y$ of $j$-closed
elements $x$, $y$ is not necessarily $j$-closed, we can still compute
the meet as $j(x\land y)$. This discussion generalises smoothly to
arbitrarily-sized [[least upper bounds]] and [[greatest lower bounds]]
in $P^j$.

```agda
  close-lub-is-lub
    : ∀ {ℓ} {I : Type ℓ} {F : I → j.Fixset} {lub : Ob}
    → is-lub P (fst ∘ F) lub
    → is-lub j.Fixposet F (j.unit lub)
  close-lub-is-lub {F = F} {lub} isl .fam≤lub i =
    F i .fst            =⟨ F i .snd ⟩
    j.close (F i .fst)  ≤⟨ j.pres-≤ (isl .fam≤lub i) ⟩
    j.close lub         ≤∎
  close-lub-is-lub {lub = lub} isl .least (ub' , closed) below =
    j.close lub  ≤⟨ j.pres-≤ (isl .least ub' below) ⟩
    j.close ub'  =˘⟨ closed ⟩
    ub'          ≤∎

  close-glb-is-glb
    : ∀ {ℓ} {I : Type ℓ} {F : I → j.Fixset} {glb : Ob}
    → is-glb P (fst ∘ F) glb
    → is-glb j.Fixposet F (j.unit glb)
  close-glb-is-glb {F = F} {glb} isg .glb≤fam i =
    j.close glb        ≤⟨ j.pres-≤ (isg .glb≤fam i) ⟩
    j.close (F i .fst) =˘⟨ F i .snd ⟩
    F i .fst           ≤∎
  close-glb-is-glb {glb = glb} isg .greatest (lb' , closed) above =
    lb'         =⟨ closed ⟩
    j.close lb' ≤⟨ j.pres-≤ (isg .greatest lb' above) ⟩
    j.close glb ≤∎
```

<!--
```agda
  open is-suplattice

  Fixset-is-suplattice : is-suplattice P → is-suplattice (Closure.Fixposet j)
  Fixset-is-suplattice supl .⋃ F = Closure.unit j (supl .⋃ (fst ∘ F))
  Fixset-is-suplattice supl .⋃-joins F = close-lub-is-lub (supl .⋃-joins _)
```
-->
