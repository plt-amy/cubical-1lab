---
description: We compute the composite of two adjunctions.
---
<!--
```agda
{-# OPTIONS --lossy-unification #-}
open import Cat.Functor.Adjoint
open import Cat.Prelude

import Cat.Functor.Reasoning
import Cat.Reasoning
```
-->

```agda
module Cat.Functor.Adjoint.Compose
```

# Composition of adjunctions {defines="composition-of-adjunctions adjunctions-compose"}

Suppose we have four functors $F \dashv G$ and $L \dashv R$, such that
they "fit together", i.e. the composites $LF$ and $GR$ both exist. What
can we say about their composites? The hope is that they would again be
adjoints, and this is indeed the case.

We prove this here by explicitly exhibiting the adjunction natural
transformations and the triangle identities, which is definitely
suboptimal for readability, but is the most efficient choice in terms of
the resulting Agda program.

```agda
    {o ℓ o₂ ℓ₂ o₃ ℓ₃}
    {A : Precategory o ℓ} {B : Precategory o₂ ℓ₂}
    {C : Precategory o₃ ℓ₃}
    {F : Functor A B} {G : Functor B A}
    {L : Functor B C} {R : Functor C B}
    (F⊣G : F ⊣ G)
    (L⊣R : L ⊣ R)
  where
```

<!--
```agda
private
  module fg = _⊣_ F⊣G
  module lr = _⊣_ L⊣R
  module A = Cat.Reasoning A
  module B = Cat.Reasoning B
  module C = Cat.Reasoning C
  module F = Cat.Functor.Reasoning F
  module G = Cat.Functor.Reasoning G
  module L = Cat.Functor.Reasoning L
  module R = Cat.Functor.Reasoning R
  open _⊣_
  open _=>_
  module LF = Functor (L F∘ F)
  module GR = Functor (G F∘ R)
```
-->

```agda
LF⊣GR : (L F∘ F) ⊣ (G F∘ R)
LF⊣GR .unit .η x          = G.₁ (lr.η _) A.∘ fg.η _
LF⊣GR .counit .η x        = lr.ε _ C.∘ L.₁ (fg.ε _)

LF⊣GR .unit .is-natural x y f =
  (G.₁ (lr.η _) A.∘ fg.η _) A.∘ f                ≡⟨ A.pullr (fg.unit.is-natural _ _ _) ⟩
  G.₁ (lr.η _) A.∘ G.₁ (F.₁ f) A.∘ fg.η _        ≡⟨ A.pulll (sym (G.F-∘ _ _)) ⟩
  G.₁ ⌜ lr.η _ B.∘ F.₁ f ⌝ A.∘ fg.η _            ≡⟨ ap! (lr.unit.is-natural _ _ _) ⟩
  G.₁ (R.₁ (L.₁ (F.₁ f)) B.∘ lr.η _) A.∘ fg.η _  ≡⟨ A.pushl (G.F-∘ _ _) ⟩
  GR.₁ (LF.₁ f) A.∘ G.₁ (lr.η _) A.∘ (fg.η _)    ∎

LF⊣GR .counit .is-natural x y f =
  (lr.ε _ C.∘ L.₁ (fg.ε _)) C.∘ LF.₁ (GR.₁ f) ≡⟨ C.pullr (sym (L.F-∘ _ _)) ⟩
  lr.ε _ C.∘ L.₁ ⌜ fg.ε _ B.∘ F.₁ (GR.₁ f) ⌝  ≡⟨ ap! (fg.counit.is-natural _ _ _) ⟩
  lr.ε _ C.∘ ⌜ L.₁ (R.F₁ f B.∘ fg.ε _) ⌝      ≡⟨ ap! (L.F-∘ _ _) ⟩
  lr.ε _ C.∘ L.₁ (R.F₁ f) C.∘ L.₁ (fg.ε _)    ≡⟨ C.extendl (lr.counit.is-natural _ _ _) ⟩
  f C.∘ lr.ε _ C.∘ L.₁ (fg.ε _)               ∎

LF⊣GR .zig =
  (lr.ε _ C.∘ L.₁ (fg.ε _)) C.∘ ⌜ LF.₁ (G.₁ (lr.η _) A.∘ fg.η _) ⌝ ≡⟨ C.extendr (ap! (LF.F-∘ _ _) ∙ L.extendl (fg.counit.is-natural _ _ _)) ⟩
  (lr.ε _ C.∘ L.₁ (lr.η _)) C.∘ (L.₁ (fg.ε _) C.∘ LF.₁ (fg.η _))   ≡⟨ C.elimr (L.annihilate fg.zig) ⟩
  lr.ε _ C.∘ L.₁ (lr.η _)                                          ≡⟨ lr.zig ⟩
  C.id                                                             ∎

LF⊣GR .zag =
  GR.₁ (lr.ε _ C.∘ L.₁ (fg.ε _)) A.∘ G.₁ (lr.η _) A.∘ fg.η _ ≡⟨ A.pulll (G.collapse (B.pushl (R.F-∘ _ _) ∙ ap₂ B._∘_ refl (sym (lr.unit.is-natural _ _ _)))) ⟩
  G.₁ ⌜ R.₁ (lr.ε _) B.∘ lr.η _ B.∘ fg.ε _ ⌝ A.∘ fg.η _      ≡⟨ ap! (B.cancell lr.zag) ⟩
  G.₁ (fg.ε _) A.∘ fg.η _                                    ≡⟨ fg.zag ⟩
  A.id                                                       ∎
```
