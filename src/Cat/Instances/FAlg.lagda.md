```agda
open import Cat.Prelude
open import Cat.Diagram.Initial
open import Cat.Displayed.Total
open import Cat.Displayed.Base
open Total-hom

import Cat.Reasoning

module Cat.Instances.FAlg where
```
```agda

module _ {o ℓ} {C : Precategory o ℓ} (F : Functor C C) where

  open Cat.Reasoning C
  open Functor F
  open Displayed

  FAlg : Displayed C _ _
  Ob[ FAlg ] A          = Hom (F₀ A) A
  Hom[ FAlg ] h α β     = h ∘ α ≡ β ∘ F₁ h
  Hom[ FAlg ]-set _ _ _ = hlevel!
  FAlg .id′             = idl _ ∙ intror F-id
  FAlg ._∘′_ p q        = pullr q ∙ extendl p ∙ ap (_ ∘_) (sym (F-∘ _ _))
  FAlg .idr′ _          = prop!
  FAlg .idl′ _          = prop!
  FAlg .assoc′ _ _ _    = prop!

```

```agda

  FAlgebras : Precategory _ _
  FAlgebras = ∫ FAlg

  module FAlgebras = Cat.Reasoning FAlgebras

  lambek : ∀ (i : FAlgebras.Ob) → is-initial FAlgebras i → is-invertible (i .snd)
  lambek (I , i) init = make-invertible (j .hom) p q
    where
      j : FAlgebras.Hom (I , i) (F₀ I , F₁ i)
      j = init (F₀ I , F₁ i) .centre

      i' : FAlgebras.Hom (F₀ I , F₁ i) (I , i)
      i' .hom = i
      i' .preserves = refl
      
      p = ap hom (is-contr→is-prop (init (I , i)) (i' FAlgebras.∘ j) FAlgebras.id)
      q = (j .hom ∘ i)       ≡⟨ j .preserves ⟩
          F₁ i ∘ F₁ (j .hom) ≡˘⟨ F-∘ _ _ ⟩
          F₁ (i ∘ j .hom)    ≡⟨ ap F₁ p ⟩
          F₁ id              ≡⟨ F-id ⟩
          id ∎
```
