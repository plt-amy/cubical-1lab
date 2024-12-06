<!--
```agda
open import Cat.Functor.Adjoint
open import Cat.Functor.Base
open import Cat.Prelude

import Cat.Reasoning
```
-->

```agda
module Cat.Diagram.Comonad where
```

<!--
```agda
open Functor
open _=>_

module _ {o ℓ} {C : Precategory o ℓ} where
  open Precategory C
```
-->

# Comonads

A **comonad on a category** $\cC$ is dual to a [monad] on $\cC$; instead
of a unit $\Id \To M$ and multiplication $(M \circ M) \To M$, we have a
counit $W \To \Id$ and comultiplication $W \To (W \circ W)$. More
generally, we can define what it means to equip a *fixed* functor $W$
with the structure of a comonad.

[monad]: Cat.Diagram.Monad.html

```agda
  record Comonad-on (W : Functor C C) : Type (o ⊔ ℓ) where
    field
      counit : W => Id
      comult : W => (W F∘ W)
```

<!--
```agda
    module counit = _=>_ counit renaming (η to ε)
    module comult = _=>_ comult renaming (η to δ)

    open Functor W renaming (F₀ to W₀ ; F₁ to W₁ ; F-id to W-id ; F-∘ to W-∘) public
    open counit using (ε) public
    open comult using (δ) public
```
-->

We also have equations governing the counit and comultiplication.
Unsurprisingly, these are dual to the equations of a monad.

```agda
    field
      δ-idl   : ∀ {x} → W₁ (ε x) ∘ δ x ≡ id
      δ-idr   : ∀ {x} → ε (W₀ x) ∘ δ x ≡ id
      δ-assoc : ∀ {x} → W₁ (δ x) ∘ δ x ≡ δ (W₀ x) ∘ δ x
```
