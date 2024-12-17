---
description: We define the monad associated to an adjunction.
---
<!--
```agda
open import Cat.Functor.Adjoint
open import Cat.Diagram.Monad
open import Cat.Prelude

open Monad-on
open Functor
open _=>_
```
-->

# The monad from an adjunction {defines="monad-from-adjunction"}

```agda
module
  Cat.Functor.Adjoint.Monad
  {o₁ h₁ o₂ h₂ : _}
  {C : Precategory o₁ h₁}
  {D : Precategory o₂ h₂}
  {L : Functor C D} {R : Functor D C}
  (L⊣R : L ⊣ R)
  where
```

<!--
```agda
private
  module C = Precategory C
  module D = Precategory D
  module L = Functor L
  module R = Functor R
  module adj = _⊣_ L⊣R
```
-->

Every adjunction $L \dashv R$ gives rise to a monad, where the
underlying functor is $R \circ L$.

```agda
Adjunction→Monad : Monad-on (R F∘ L)
```

The unit of the monad is just adjunction monad, and the multiplication
comes from the counit.

```agda
Adjunction→Monad .unit = adj.unit
Adjunction→Monad .mult = NT (λ x → R.₁ (adj.ε (L.₀ x))) λ x y f →
  R.₁ (adj.ε (L.₀ y)) C.∘ R.₁ (L.₁ (R.₁ (L.₁ f))) ≡⟨ sym (R.F-∘ _ _) ⟩
  R.₁ (adj.ε (L.₀ y) D.∘ L.₁ (R.₁ (L.₁ f)))       ≡⟨ ap R.₁ (adj.counit.is-natural _ _ _) ⟩
  R.₁ (L.₁ f D.∘ adj.ε (L.₀ x))                   ≡⟨ R.F-∘ _ _ ⟩
  _                                               ∎
```

The monad laws follow from the zig-zag identities. In fact, the
`right-ident`{.Agda}ity law is exactly the `zag`{.Agda ident="adj.zag"}
identity.

```agda
Adjunction→Monad .μ-unitl {x} = adj.zag
```

The others are slightly more involved.

```agda
Adjunction→Monad .μ-unitr {x} = path where abstract
  path : R.₁ (adj.ε (L.F₀ x)) C.∘ R.₁ (L.₁ (adj.η x)) ≡ C.id
  path =
    R.₁ (adj.ε _) C.∘ R.₁ (L.₁ (adj.η _)) ≡⟨ sym (R.F-∘ _ _) ⟩
    R.₁ (adj.ε _ D.∘ L.₁ (adj.η _))       ≡⟨ ap R.₁ adj.zig ⟩
    R.₁ D.id                              ≡⟨ R.F-id ⟩
    C.id                                  ∎

Adjunction→Monad .μ-assoc {x} = path where abstract
  path : R.₁ (adj.ε _) C.∘ R.₁ (L.₁ (R.₁ (adj.ε (L.F₀ x))))
       ≡ R.₁ (adj.ε _) C.∘ R.₁ (adj.ε (L .F₀ (R.F₀ (L.F₀ x))))
  path =
    R.₁ (adj.ε _) C.∘ R.₁ (L.₁ (R.₁ (adj.ε _)))   ≡⟨ sym (R.F-∘ _ _) ⟩
    R.₁ (adj.ε _ D.∘ L.₁ (R.₁ (adj.ε _)))         ≡⟨ ap R.₁ (adj.counit.is-natural _ _ _) ⟩
    R.₁ (adj.ε _ D.∘ adj.ε _)                     ≡⟨ R.F-∘ _ _ ⟩
    R.₁ (adj.ε _) C.∘ R.₁ (adj.ε _)               ∎
```
