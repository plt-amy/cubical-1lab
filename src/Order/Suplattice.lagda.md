<!--
```agda
open import 1Lab.Prelude using (_∘_)

open import Cat.Prelude

open import Data.Power

open import Meta.Idiom

open import Order.Diagram.Lub.Reasoning
open import Order.Instances.Pointwise
open import Order.Diagram.Lub.Subset
open import Order.Diagram.Lub
open import Order.Subposet
open import Order.Base hiding (pres-≤)

import Order.Reasoning as Pos
```
-->

```agda
module Order.Suplattice where
```

# Suplattices {defines="suplattice"}

A **suplattice** is a [[lattice]] that admits all [[least upper
bounds]]. By the logic of [[complete lattices]], this means that it
*also* admits all [[greatest lower bounds]]. It would, therefore, stand
to reason to have a single name for both concepts, and differentiate
entirely on the basis of what *category* they fit in: however, we prefer
to have the disambiguation also in the names.

```agda
record is-suplattice {o ℓ} (P : Poset o ℓ) : Type (lsuc o ⊔ ℓ) where
  field
    ⋃       : {I : Type o} (F : I → ⌞ P ⌟) → ⌞ P ⌟
    ⋃-joins : ∀ {I} F → is-lub P F (⋃ {I} F)

  open Lubs P ⋃-joins public
  open Join-subsets P ⋃-joins public
```

The morphisms between suplattices are the **cocontinuous functions**:
the monotone maps which additionally send (arbitrary!) joins to joins.

```agda
record
  is-cocontinuous
    {o ℓ ℓ'} {P : Poset o ℓ} {Q : Poset o ℓ'}
    (f : Monotone P Q) (Ps : is-suplattice P) (Qs : is-suplattice Q)
    : Type (lsuc o ⊔ ℓ')
  where
```

<!--
```agda
  private
    module Ps = is-suplattice Ps
    module Qs = is-suplattice Qs
    module Q = Poset Q

  open Monotone
```
-->

```agda
  field
    ⋃-≤ : {I : Type o} (k : I → ⌞ P ⌟) → f # Ps.⋃ k Q.≤ Qs.⋃ (apply f ∘ k)
```

<!--
```agda
  abstract
    pres-⋃ : {I : Type o} (F : I → ⌞ P ⌟) → f # Ps.⋃ F ≡ Qs.⋃ (apply f ∘ F)
    pres-⋃ k = Q.≤-antisym (⋃-≤ k) (Qs.⋃-universal _ λ i → f .pres-≤ (Ps.⋃-inj i))
```
-->
