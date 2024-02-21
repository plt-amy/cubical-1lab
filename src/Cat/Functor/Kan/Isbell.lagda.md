---
description: |
  We use the construction of Realisation Nerve adjunction to obtain an adjunction between Presheaves and Copresheaves
---

<!--
```agda
open import Cat.Functor.Adjoint
open import Cat.Prelude
```
-->

```agda
module Cat.Functor.Kan.Isbell where
```

<!--
```agda
open import Cat.Functor.Base
open import Cat.Functor.Kan.Nerve
open import Cat.Functor.Hom
```
-->
# Isbell duality
## Copresheaves
First we define co-presheaves, where presheaves are the functorial category from a category $\cC ^{op}$ to $\Sets$,
co-presheaves are the opposite of the functor category from $\cC$ to $\Sets$.
```agda
CoPSh : ∀ κ {o ℓ} → Precategory o ℓ → Precategory _ _
CoPSh κ C = Cat[ C , Sets κ ] ^op
```

Co-presheaves come with their own variant of Yoneda embedding, the co-Yoneda embedding $\mathbb{z}$.
The co-Yoneda embedding is just the opposite of the Yoneda embedding on the opposite category.
```agda
module _ {o κ} (C : Precategory o κ) where
  z : Functor C (CoPSh κ C)
  z = Functor.op (よ (C ^op))
```
<!--
```agda
open import Cat.Diagram.Colimit.Base
open import Cat.Diagram.Limit.Base
```
-->

We will require the fact that the opposite of a complete category is cocomplete, this is a direct consequence of duality for diagrams.
```agda
module _ {o}{l}{x} {y} {C : Precategory o l} where

  open import Cat.Diagram.Duals

  is-complete-is-cocomplete-op : is-complete x y C ->  is-cocomplete x y (C ^op)
  is-complete-is-cocomplete-op isCompl F = Co-limit→Colimit _ (isCompl _)
```

Next we prove that Copresheaves are co-complete this is a direct consequence of opposites swapping completeness for cocompletess,
together with completeness of the functor category and Sets.
```agda
module _ {o ℓ} (C : Precategory o ℓ) where

  open import Cat.Instances.Functor.Limits
  open import Cat.Instances.Sets.Complete
  
  CoPSh-cocomplete : ∀ κ → is-cocomplete κ κ (CoPSh κ C)
  CoPSh-cocomplete κ = is-complete-is-cocomplete-op {C = Cat[ C , Sets κ ]} (Functor-cat-is-complete (Sets-is-complete {o = κ}))
```

## Isbell duality itself

As a next step we define half of the adjunction going from Co-presheaves to presheaves. This is the nerve of the Co-yoneda embedding.
```agda
Spec : ∀ κ (C : Precategory κ κ) → Functor (CoPSh κ C) (PSh κ C)
Spec κ C = Nerve (z _)
```

Now the Isbell duality is just a wrapping of the abstract non-sense coming from the Realisation⊣Nerve duality for co-presheaves as they are co-complete.
```agda
-- agda refused to figure out implicits quickly so we help it
IsbellDuality : ∀ κ  {C : Precategory κ κ} -> _ ⊣ Spec _ C
IsbellDuality κ {C} = Realisation⊣Nerve {C = C} {D = CoPSh κ C} (z C) (CoPSh-cocomplete _ _)
```
<!-- [TODO: Timo, 18/12/2023] add the alternate Conerve Corealisation construction and characterize them in terms
  of the commutative triangle in Lemma 2.2 from [Isbellandreflexive].

  [Isbellandreflexive]: http://www.tac.mta.ca/tac/volumes/36/12/36-12abs.html
  -->

