<!--
```agda
open import Cat.Diagram.Comonad
open import Cat.Displayed.Total
open import Cat.Functor.Adjoint
open import Cat.Displayed.Base
open import Cat.Prelude

import Cat.Reasoning

open Total-hom
open Functor
open _=>_
open _⊣_
```
-->

```agda
module Cat.Instances.Coalgebras where
```

# Coalgebras over a comonad {defines="coalgebra category-of-coalgebras"}

<!--
```agda
module _ {o ℓ} {C : Precategory o ℓ} {W : Functor C C} (cm : Comonad-on W) where
  open Cat.Reasoning C
  private module W = Comonad-on cm
  open W
```
-->

```agda
  record Coalgebra-on (A : Ob) : Type (o ⊔ ℓ) where
    no-eta-equality

    field
      ρ        : Hom A (W₀ A)
      ρ-counit : W.ε A ∘ ρ ≡ id
      ρ-comult : W₁ ρ ∘ ρ       ≡ δ A ∘ ρ
```

<!--
```agda
  open Coalgebra-on
  module _ where
    open Displayed
```
-->

```agda
    is-coalgebra-hom : ∀ {x y} (h : Hom x y) → Coalgebra-on x → Coalgebra-on y → Type _
    is-coalgebra-hom f α β = W₁ f ∘ α .ρ ≡ β .ρ ∘ f

    Coalgebras-over : Displayed C (o ⊔ ℓ) ℓ
    Coalgebras-over .Ob[_]            = Coalgebra-on
    Coalgebras-over .Hom[_]           = is-coalgebra-hom
    Coalgebras-over .Hom[_]-set f α β = hlevel 2

    Coalgebras-over .id'      = eliml W-id ∙ intror refl
    Coalgebras-over ._∘'_ p q = pushl (W-∘ _ _) ·· ap (W₁ _ ∘_) q ·· extendl p

    Coalgebras-over .idr' f'         = prop!
    Coalgebras-over .idl' f'         = prop!
    Coalgebras-over .assoc' f' g' h' = prop!

  Coalgebras : Precategory (o ⊔ ℓ) ℓ
  Coalgebras = ∫ Coalgebras-over
```

<!--
```agda
  module Coalgebras = Cat.Reasoning Coalgebras

module _ {o ℓ} {C : Precategory o ℓ} {F : Functor C C} {W : Comonad-on F} where instance
  Extensional-coalgebra-hom
    : ∀ {ℓr} {x y} ⦃ _ : Extensional (C .Precategory.Hom (x .fst) (y .fst)) ℓr ⦄
    → Extensional (Coalgebras.Hom W x y) ℓr
  Extensional-coalgebra-hom ⦃ e ⦄ = injection→extensional! (λ p → total-hom-path (Coalgebras-over W) p prop!) e

module _ {o ℓ} {C : Precategory o ℓ} {F : Functor C C} (W : Comonad-on F) where
  open Cat.Reasoning C
  private module W = Comonad-on W
  open Coalgebra-on
  open W
```
-->

## Cofree coalgebras {defines="cofree-coalgebra"}

```agda
  Cofree-coalgebra : Ob → Coalgebras.Ob W
  Cofree-coalgebra A .fst = W₀ A
  Cofree-coalgebra A .snd .ρ = δ _
  Cofree-coalgebra A .snd .ρ-counit = δ-idr
  Cofree-coalgebra A .snd .ρ-comult = δ-assoc

  Cofree : Functor C (Coalgebras W)
  Cofree .F₀ = Cofree-coalgebra

  Cofree .F₁ h .hom       = W₁ h
  Cofree .F₁ h .preserves = sym (comult.is-natural _ _ h)

  Cofree .F-id    = ext W-id
  Cofree .F-∘ f g = ext (W-∘ _ _)
```

```agda
  Forget⊣Cofree : πᶠ (Coalgebras-over W) ⊣ Cofree
  Forget⊣Cofree .unit .η (x , α) .hom       = α .ρ
  Forget⊣Cofree .unit .η (x , α) .preserves = α .ρ-comult
  Forget⊣Cofree .unit .is-natural x y f = ext (sym (f .preserves))

  Forget⊣Cofree .counit .η x              = W.ε x
  Forget⊣Cofree .counit .is-natural x y f = W.counit.is-natural _ _ _

  Forget⊣Cofree .zig {A , α} = α .ρ-counit
  Forget⊣Cofree .zag         = ext δ-idl
```

<!--
```agda
  to-cofree-hom
    : ∀ {X Y} → Hom (X .fst) Y → Coalgebras.Hom W X (Cofree-coalgebra Y)
  to-cofree-hom f = L-adjunct Forget⊣Cofree f
```
-->
