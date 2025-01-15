<!--
```agda
open import Cat.Functor.Subcategory
open import Cat.Prelude

open import Order.Instances.Pointwise.Diagrams
open import Order.Diagram.Lub.Reasoning
open import Order.Instances.Pointwise
open import Order.Semilattice.Join
open import Order.Semilattice.Meet
open import Order.Instances.Props
open import Order.Diagram.Meet
open import Order.Diagram.Lub
open import Order.Diagram.Top
open import Order.Lattice
open import Order.Base

import Cat.Reasoning

import Order.Diagram.Meet.Reasoning as Meets
import Order.Reasoning
```
-->

```agda
module Order.Frame where
```

# Frames {defines="frame"}

A **frame** is a [[lattice]] with finite [[meets]]^[So, in addition to the $x
\cap y$ operation, we have a top element] and arbitrary [[joins]] satisfying
the **infinite distributive law**

$$
x \cap \bigcup_i f(i) = \bigcup_i (x \cap f(i))
$$.

In the study of frames, for simplicity, we assume [[propositional
resizing]]: that way, it suffices for a frame $A$ to have joins of
$\cJ$-indexed families, for $\cJ$ an arbitrary type in the same universe
as $A$, to have joins for arbitrary subsets of $A$.

```agda
record is-frame {o ℓ} (P : Poset o ℓ) : Type (lsuc o ⊔ ℓ) where
  no-eta-equality
  open Poset P
  field
    _∩_     : Ob → Ob → Ob
    ∩-meets : ∀ x y → is-meet P x y (x ∩ y)

    has-top : Top P

    ⋃       : ∀ {I : Type o} (k : I → Ob) → Ob
    ⋃-lubs  : ∀ {I : Type o} (k : I → Ob) → is-lub P k (⋃ k)

    ⋃-distribl : ∀ {I} x (f : I → Ob) → x ∩ ⋃ f ≡ ⋃ λ i → x ∩ f i
```

We have explicitly required that a frame be a [[meet-semilattice]], but
it's worth explicitly pointing out that the infinitary join operation
can also be used for more mundane purposes: By taking a join over the
type of booleans (and over the empty type), we can show that all frames
are also [[join-semilattices]].

<!--
```agda
  infixr 25 _∩_

  module is-lubs {I} {k : I → Ob} = is-lub (⋃-lubs k)

  open Meets ∩-meets public
  open Top has-top using (top; !) public
  open Lubs P ⋃-lubs public

  has-meet-slat : is-meet-semilattice P
  has-meet-slat .is-meet-semilattice._∩_ = _∩_
  has-meet-slat .is-meet-semilattice.∩-meets = ∩-meets
  has-meet-slat .is-meet-semilattice.has-top = has-top

  has-join-slat : is-join-semilattice P
  has-join-slat .is-join-semilattice._∪_ = _∪_
  has-join-slat .is-join-semilattice.∪-joins = ∪-joins
  has-join-slat .is-join-semilattice.has-bottom = has-bottom

  has-lattice : is-lattice P
  has-lattice .is-lattice._∩_ = _∩_
  has-lattice .is-lattice.∩-meets = ∩-meets
  has-lattice .is-lattice._∪_ = _∪_
  has-lattice .is-lattice.∪-joins = ∪-joins
  has-lattice .is-lattice.has-top = has-top
  has-lattice .is-lattice.has-bottom = has-bottom

private variable
  o ℓ o' ℓ' : Level
  P Q R : Poset o ℓ

abstract
  is-frame-is-prop : is-prop (is-frame P)
  is-frame-is-prop {P = P} p q = path where
    open Order.Diagram.Top P using (H-Level-Top)

    module p = is-frame p
    module q = is-frame q
    open is-frame

    meetp : ∀ x y → x p.∩ y ≡ x q.∩ y
    meetp x y = meet-unique (p.∩-meets x y) (q.∩-meets x y)

    lubp : ∀ {I} (k : I → ⌞ P ⌟) → p.⋃ k ≡ q.⋃ k
    lubp k = lub-unique (p.⋃-lubs k) (q.⋃-lubs k)

    path : p ≡ q
    path i ._∩_     x y = meetp x y i
    path i .∩-meets x y = is-prop→pathp (λ i → hlevel {T = is-meet P x y (meetp x y i)} 1) (p.∩-meets x y) (q.∩-meets x y) i
    path i .has-top    = hlevel {T = Top P} 1 p.has-top q.has-top i
    path i .⋃ k        = lubp k i
    path i .⋃-lubs k = is-prop→pathp (λ i → hlevel {T = is-lub P k (lubp k i)} 1) (p.⋃-lubs k) (q.⋃-lubs k) i
    path i .⋃-distribl x f j = is-set→squarep (λ _ _ → Poset.Ob-is-set P)
      (λ i → meetp x (lubp f i) i)
      (p.⋃-distribl x f) (q.⋃-distribl x f)
      (λ i → lubp (λ e → meetp x (f e) i) i)
      i j

instance
  H-Level-is-frame : ∀ {n} → H-Level (is-frame P) (suc n)
  H-Level-is-frame = prop-instance is-frame-is-prop
```
-->

Of course, a frame is not just a lattice, but a *complete* lattice.
Since the infinite distributive law says exactly that "meet with $x$"
preserves joins, this implies that it has a right adjoint, so frames are
also complete [[Heyting algebras]]. Once again, the difference in naming
reflects the morphisms we will consider these structures under: A
**frame homomorphism** is a [[monotone map]] which preserves the finite
meets and the infinitary joins, but not necessarily the infinitary meets
(or the Heyting implication).

<!-- [TODO: Reed M, 10/01/2024] Prove that all joins => heyting algebras + link to proof here -->

Since meets and joins are defined by a universal property, and we have
assumed that homomorphisms are *a priori* monotone, it suffices to show
the following inequalities:

* For every $x, y : P$, we have $f x \cap f y \leq f (x \cap y)$;
* $\top \leq f(\top)$;
* and finally, for every family $k : I \to P$, we have $f (\bigcup k) \leq \bigcup (f \circ k)$

```agda
record
  is-frame-hom
    {P : Poset o ℓ} {Q : Poset o ℓ'}
    (f : Monotone P Q) (P-frame : is-frame P) (Q-frame : is-frame Q)
    : Type (lsuc o ⊔ ℓ') where
```

<!--
```agda
  private
    module P = Poset P
    module Pᶠ = is-frame P-frame
    module Q = Order.Reasoning Q
    module Qᶠ = is-frame Q-frame
    open is-lub
```
-->

```agda
  field
    ∩-≤   : ∀ x y → (f # x) Qᶠ.∩ (f # y) Q.≤ f # (x Pᶠ.∩ y)
    top-≤ : Qᶠ.top Q.≤ f # Pᶠ.top
    ⋃-≤   : ∀ {I : Type o} (k : I → ⌞ P ⌟) → (f # Pᶠ.⋃ k) Q.≤ Qᶠ.⋃ (apply f ⊙ k)
```

If $f$ is a frame homomorphism, then it is also a homomorphism of [[meet
semilattices]].

```agda
  has-meet-slat-hom : is-meet-slat-hom f Pᶠ.has-meet-slat Qᶠ.has-meet-slat
  has-meet-slat-hom .is-meet-slat-hom.∩-≤ = ∩-≤
  has-meet-slat-hom .is-meet-slat-hom.top-≤ = top-≤

  open is-meet-slat-hom has-meet-slat-hom hiding (∩-≤; top-≤) public
```

Furthermore, we can actually show from the inequality required above
that $f$ preserves all joins up to equality.

```agda
  pres-⋃ : ∀ {I : Type o} (k : I → ⌞ P ⌟) → (f # Pᶠ.⋃ k) ≡ Qᶠ.⋃ (apply f ⊙ k)
  pres-⋃ k =
    Q.≤-antisym
      (⋃-≤ k)
      (Qᶠ.⋃-universal _ (λ i → f .pres-≤ (Pᶠ.⋃-inj i)))

  pres-lubs
    : ∀ {I : Type o} {k : I → ⌞ P ⌟} {l}
    → is-lub P k l
    → is-lub Q (apply f ⊙ k) (f # l)
  pres-lubs {k = k} {l = l} l-lub .fam≤lub i = f .pres-≤ (l-lub .fam≤lub i)
  pres-lubs {k = k} {l = l} l-lub .least ub p =
    f # l              Q.≤⟨ f .pres-≤ (l-lub .least _ Pᶠ.⋃-inj) ⟩
    f # Pᶠ.⋃ k         Q.≤⟨ ⋃-≤ k ⟩
    Qᶠ.⋃ (apply f ⊙ k) Q.≤⟨ Qᶠ.⋃-universal ub p ⟩
    ub                 Q.≤∎
```

As a corollary, $f$ is also a homomorphism of the underlying [[*join*
semilattices]].

```agda
  opaque
    unfolding Lubs.∪-joins Lubs.has-bottom

    has-join-slat-hom : is-join-slat-hom f Pᶠ.has-join-slat Qᶠ.has-join-slat
    has-join-slat-hom .is-join-slat-hom.∪-≤ x y =
      Q.≤-trans (⋃-≤ _) $ Qᶠ.⋃-universal _ λ where
        (lift true) → Qᶠ.⋃-inj (lift true)
        (lift false) → Qᶠ.⋃-inj (lift false)
    has-join-slat-hom .is-join-slat-hom.bot-≤ =
      Q.≤-trans (⋃-≤ _) (Qᶠ.⋃-universal _ (λ ()))

  open is-join-slat-hom has-join-slat-hom public

open is-frame-hom
```

<!--
```agda
unquoteDecl H-Level-is-frame-hom = declare-record-hlevel 1 H-Level-is-frame-hom (quote is-frame-hom)
```
-->

Clearly, the identity function is a frame homomorphism.

```agda
id-frame-hom
  : ∀ (Pᶠ : is-frame P)
  → is-frame-hom idₘ Pᶠ Pᶠ
id-frame-hom {P = P} Pᶠ .∩-≤ x y =
  Poset.≤-refl P
id-frame-hom {P = P} Pᶠ .top-≤ =
  Poset.≤-refl P
id-frame-hom {P = P} Pᶠ .⋃-≤ k =
  Poset.≤-refl P
```

Furthermore, frame homomorphisms are closed under composition.

```agda
∘-frame-hom
  : ∀ {Pₗ Qₗ Rₗ} {f : Monotone Q R} {g : Monotone P Q}
  → is-frame-hom f Qₗ Rₗ
  → is-frame-hom g Pₗ Qₗ
  → is-frame-hom (f ∘ₘ g) Pₗ Rₗ
∘-frame-hom {R = R} {f = f} {g = g} f-pres g-pres .∩-≤ x y =
  R .Poset.≤-trans (f-pres .∩-≤ (g # x) (g # y)) (f .pres-≤ (g-pres .∩-≤ x y))
∘-frame-hom {R = R} {f = f} {g = g} f-pres g-pres .top-≤ =
  R .Poset.≤-trans (f-pres .top-≤) (f .pres-≤ (g-pres .top-≤))
∘-frame-hom {R = R} {f = f} {g = g} f-pres g-pres .⋃-≤ k =
  R .Poset.≤-trans (f .pres-≤ (g-pres .⋃-≤ k)) (f-pres .⋃-≤ (λ i → g # k i))
```

This means that we can define the category of frames as a [[subcategory]]
of the category of posets.

```agda
Frame-subcat : ∀ o ℓ → Subcat (Posets o ℓ) _ _
Frame-subcat o ℓ .Subcat.is-ob = is-frame
Frame-subcat o ℓ .Subcat.is-hom = is-frame-hom
Frame-subcat o ℓ .Subcat.is-hom-prop _ _ _ = hlevel 1
Frame-subcat o ℓ .Subcat.is-hom-id = id-frame-hom
Frame-subcat o ℓ .Subcat.is-hom-∘ = ∘-frame-hom

Frames : ∀ o ℓ → Precategory _ _
Frames o ℓ = Subcategory (Frame-subcat o ℓ)

module Frames {o} {ℓ} = Cat.Reasoning (Frames o ℓ)

Frame : ∀ o ℓ → Type _
Frame o ℓ = Frames.Ob {o} {ℓ}
```

## Power sets as frames

A canonical source of frames are power sets: The power set of any type
is a frame, because it is a complete lattice satisfying the infinite
distributive law; This can be seen by some computation, as is done
below.

```agda
open is-frame
open is-meet-semilattice

Power-frame : ∀ {ℓ} (A : Type ℓ) → Frame ℓ ℓ
Power-frame {ℓ = ℓ} A .fst = Subsets A
Power-frame A .snd ._∩_ P Q i = P i ∧Ω Q i
Power-frame A .snd .∩-meets P Q =
  is-meet-pointwise λ _ → Props-has-meets (P _) (Q _)
Power-frame A .snd .has-top =
  has-top-pointwise λ _ → Props-has-top
Power-frame A .snd .⋃ k i = ∃Ω _ (λ j → k j i)
Power-frame A .snd .⋃-lubs k = is-lub-pointwise _ _ λ _ →
  Props-has-lubs λ i → k i _
Power-frame A .snd .⋃-distribl x f = ext λ i → biimp
  (rec! λ xi j j~i → inc (j , xi , j~i))
  (rec! λ j xi j~i → xi , inc (j , j~i))
```
