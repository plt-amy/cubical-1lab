---
description: |
  We establish the theory of monadic adjunctions, and define the
  comparison functor into the Eilenberg-Moore category for the induced
  monad.
---
<!--
```agda
open import Cat.Functor.Adjoint.Monad
open import Cat.Functor.Equivalence
open import Cat.Displayed.Total
open import Cat.Functor.Adjoint
open import Cat.Diagram.Monad
open import Cat.Prelude

import Cat.Functor.Reasoning
import Cat.Reasoning

open Total-hom
open Functor
open _=>_
```
-->

# Monadic adjunctions {defines="monadic-adjunction monadic-functor monadic"}

An adjunction $F \dashv G$ between functors $F : C \to D$ and $G : D \to
C$ is _monadic_ if the induced `comparison functor`{.Agda
ident=Comparison-EM} $D \to C^{R \circ L}$ (where the right-hand side is
the `Eilenberg-Moore`{.Agda} category of the [monad of the
adjunction](Cat.Functor.Adjoint.Monad.html)) is an equivalence of
categories.

```agda
module
  Cat.Functor.Adjoint.Monadic
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
  module C = Cat.Reasoning C
  module D = Cat.Reasoning D
  module L = Cat.Functor.Reasoning L
  module R = Cat.Functor.Reasoning R
  module adj = _⊣_ L⊣R

L∘R : Monad-on _
L∘R = Adjunction→Monad L⊣R

open Monad-on L∘R

private
  module Kleisli = Cat.Reasoning (Kleisli L∘R)
  module EM = Cat.Reasoning (Eilenberg-Moore L∘R)
```
-->

The composition of `R.₁`{.Agda} with the `adjunction counit`{.Agda
ident="adj.counit.ε"} natural transformation gives `R`{.Agda} an
`Algebra`{.Agda} structure, thus extending `R` to a functor $D \to C^{L
\circ R}$.

```agda
Comparison-EM : Functor D (Eilenberg-Moore L∘R)
Comparison-EM .F₀ x = R.₀ x , alg where
  alg : Algebra-on L∘R (R.₀ x)
  alg .Algebra-on.ν = R.₁ (adj.counit.ε _)
  alg .Algebra-on.ν-unit = adj.zag
  alg .Algebra-on.ν-mult = R.weave (adj.counit.is-natural _ _ _)
```

<details>
<summary> Construction of the functorial action of `Comparison-EM`{.Agda} </summary>

```agda
Comparison-EM .F₁ x .hom = R.₁ x
Comparison-EM .F₁ x .preserves = R.weave (sym (adj.counit.is-natural _ _ _))
Comparison-EM .F-id    = ext R.F-id
Comparison-EM .F-∘ f g = ext (R.F-∘ _ _)
```
</details>

An adjunction is _monadic_ if `Comparison-EM`{.Agda} is an [equivalence of
categories], thus exhibiting $C$ as the category of $R\circ L$-algebras:

[equivalence of categories]: Cat.Functor.Equivalence.html

```agda
is-monadic : Type _
is-monadic = is-equivalence Comparison-EM
```
