<!--
```agda
open import Cat.Instances.Sets.Complete
open import Cat.Diagram.Product.Solver
open import Cat.Monoidal.Diagonals
open import Cat.Instances.Functor
open import Cat.Diagram.Terminal
open import Cat.Monoidal.Braided
open import Cat.Diagram.Product
open import Cat.Monoidal.Base
open import Cat.Prelude

import Cat.Reasoning as Cr
```
-->

```agda
module Cat.Monoidal.Instances.Cartesian where
```

# Cartesian monoidal categories {defines="cartesian-monoidal-category"}

Unlike with [[categories]] and [[bicategories]], there is no handy example
of [[monoidal category]] that is as canonical as how the collection of all
$n$-categories is an $(n+1)$-category. However, we do have _a_ certain
canonical pool of examples to draw from: all the **Cartesian monoidal
categories**, also known as _finite-products categories_.

```agda
module _
  {o ℓ} {C : Precategory o ℓ}
  (prods : ∀ A B → Product C A B) (term : Terminal C)
  where
```

<!--
```agda
  open Monoidal-category hiding (_⊗₁_)
  open Braided-monoidal
  open Symmetric-monoidal
  open Diagonals hiding (δ)
  open make-natural-iso
  open Cr C
  open Binary-products C prods
  open Terminal term
```
-->

```agda
  Cartesian-monoidal : Monoidal-category C
  Cartesian-monoidal .-⊗- = ×-functor
  Cartesian-monoidal .Unit = top
```

There's nothing much to say about this result: It's pretty much just
banging out the calculation. Our tensor product functor is the Cartesian
product functor, and the tensor unit is the [[terminal object]] (the empty
product). Associators and units are the evident maps, which are coherent
by the properties of limits. Translating this intuitive explanation to a
formal proof requires a _lot_ of calculation, however:

```agda
  Cartesian-monoidal .unitor-l = to-natural-iso ni where
    ni : make-natural-iso _ _
    ni .eta x = ⟨ ! , id ⟩
    ni .inv x = π₂
    ni .eta∘inv x = Product.unique₂ (prods _ _)
      (pulll π₁∘⟨⟩ ∙ sym (!-unique _)) (cancell π₂∘⟨⟩) (!-unique₂ _ _) (idr _)
    ni .inv∘eta x = π₂∘⟨⟩
    ni .natural x y f = Product.unique₂ (prods _ _)
      (pulll π₁∘⟨⟩ ∙ pullr π₁∘⟨⟩ ∙ idl _) (pulll π₂∘⟨⟩ ∙ cancelr π₂∘⟨⟩)
      (!-unique₂ _ _) (pulll π₂∘⟨⟩ ∙ idl f)
  Cartesian-monoidal .unitor-r = to-natural-iso ni where
    ni : make-natural-iso _ _
    ni .eta x = ⟨ id , ! ⟩
    ni .inv x = π₁
    ni .eta∘inv x = Product.unique₂ (prods _ _)
      (pulll π₁∘⟨⟩ ∙ idl _) (pulll π₂∘⟨⟩ ∙ sym (!-unique _))
      (idr _) (sym (!-unique _))
    ni .inv∘eta x = π₁∘⟨⟩
    ni .natural x y f = Product.unique₂ (prods _ _)
      (pulll π₁∘⟨⟩ ·· pullr π₁∘⟨⟩ ·· idr f)
      (pulll π₂∘⟨⟩ ·· pullr π₂∘⟨⟩ ·· idl !)
      (pulll π₁∘⟨⟩ ∙ idl f)
      (!-unique₂ _ _)
  Cartesian-monoidal .associator = to-natural-iso ni where
    ni : make-natural-iso _ _
    ni .eta x = ⟨ π₁ ∘ π₁ , ⟨ π₂ ∘ π₁ , π₂ ⟩ ⟩
    ni .inv x = ⟨ ⟨ π₁ , π₁ ∘ π₂ ⟩ , π₂ ∘ π₂ ⟩
    ni .eta∘inv x = products! prods
    ni .inv∘eta x = products! prods
    ni .natural x y f = products! prods
  Cartesian-monoidal .triangle = products! prods
  Cartesian-monoidal .pentagon = products! prods
```

Cartesian monoidal categories also inherit a lot of additional structure
from the categorical product. In particular, they are [[symmetric monoidal
categories]].

```agda
  Cartesian-symmetric : Symmetric-monoidal Cartesian-monoidal
  Cartesian-symmetric = to-symmetric-monoidal mk where
    open make-symmetric-monoidal
    mk : make-symmetric-monoidal Cartesian-monoidal
    mk .has-braiding = iso→isoⁿ
      (λ _ → invertible→iso swap swap-is-iso) swap-natural
    mk .symmetric = ⟨⟩∘ _ ∙ ap₂ ⟨_,_⟩ π₂∘⟨⟩ π₁∘⟨⟩ ∙ ⟨⟩-η
    mk .has-braiding-α→ = products! prods
```

We also have a system of [[diagonal morphisms|monoidal category with diagonals]]:

```agda
  Cartesian-diagonal : Diagonals Cartesian-monoidal
  Cartesian-diagonal .diagonals ._=>_.η A = δ
  Cartesian-diagonal .diagonals ._=>_.is-natural = δ-natural
  Cartesian-diagonal .diagonal-λ→ = ap ⟨_, id ⟩ (sym (!-unique _))
```

<!--
```agda
Setsₓ : ∀ {ℓ} → Monoidal-category (Sets ℓ)
Setsₓ = Cartesian-monoidal Sets-products Sets-terminal
```
-->
