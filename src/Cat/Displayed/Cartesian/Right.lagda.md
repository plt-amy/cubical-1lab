---
description: |
  We define right fibrations, and characterize their fibre categories.
---
<!--
```agda
open import Cat.Displayed.Cartesian.Discrete
open import Cat.Displayed.Functor
open import Cat.Displayed.Base
open import Cat.Prelude

import Cat.Displayed.Cartesian
import Cat.Displayed.Reasoning
import Cat.Displayed.Morphism
import Cat.Reasoning
```
-->

```agda
module Cat.Displayed.Cartesian.Right
  {o ℓ o' ℓ'}
  {ℬ : Precategory o ℓ}
  (ℰ : Displayed ℬ o' ℓ')
  where

open Cat.Reasoning ℬ
open Displayed ℰ
open Cat.Displayed.Cartesian ℰ
open Cat.Displayed.Morphism ℰ
open Cat.Displayed.Reasoning ℰ
```

# Right fibrations

A [[cartesian fibration]] $\cE$ is said to be a **right fibration** if every
morphism in $\cE$ is cartesian.

```agda
record Right-fibration : Type (o ⊔ ℓ ⊔ o' ⊔ ℓ') where
  no-eta-equality
  field
    is-fibration : Cartesian-fibration
    cartesian
      : ∀ {x y} {f : Hom x y}
      → ∀ {x' y'} (f' : Hom[ f ] x' y')
      → is-cartesian f f'

  open Cartesian-fibration is-fibration public
```

One notable fact is every vertical morphism of a right fibrations is
invertible. In other words, if $\cE$ is a right fibration, then all
of its [[fibre categories]] are groupoids. This follows directly from the
fact that vertical cartesian maps are invertible.

```agda
right-fibration→vertical-invertible
  : Right-fibration
  → ∀ {x} {x' x'' : Ob[ x ]} → (f' : Hom[ id ] x' x'') → is-invertible↓ f'
right-fibration→vertical-invertible rfib f' =
  vertical+cartesian→invertible (Right-fibration.cartesian rfib f')
```

More notably, this is an exact characterization of categories fibred
in groupoids! If $\cE$ is a cartesian fibration where all vertical
morphisms are invertible, then it must be a right fibration.

```agda
vertical-invertible+fibration→right-fibration
  : Cartesian-fibration
  → (∀ {x} {x' x'' : Ob[ x ]} → (f' : Hom[ id ] x' x'') → is-invertible↓ f')
  → Right-fibration
vertical-invertible+fibration→right-fibration fib vert-inv
  .Right-fibration.is-fibration = fib
vertical-invertible+fibration→right-fibration fib vert-inv
  .Right-fibration.cartesian {x = x} {f = f} {x' = x'} {y' = y'} f' = f-cart where
    open Cartesian-fibration fib
    open Cartesian-lift renaming (x' to x-lift)
```

To see this, recall that [[cartesian morphisms]] are [stable under
vertical retractions]. The cartesian lift $f^{*}$ of $f$ is obviously
cartesian, so it suffices to show that there is a vertical retraction
$x^{*} \to x'$. To construct this retraction, we shall factorize $f'$
over $f \cdot id$; which yields a vertical morphism $i^{*} : x' \to x^{*}$.
By our hypotheses, $i^{*}$ is invertible, and thus is a retraction.
What remains to be shown is that the inverse to $i^{*}$ factors
$f'$ and $f^{*}$; this follows from the factorisation of $f'$ and
the fact that $i^{*}$ is invertible.

[stable under vertical retractions]: Cat.Displayed.Cartesian.html#cartesian-vertical-retraction-stable

```agda
    x* : Ob[ x ]
    x* = has-lift f y' .x-lift

    f* : Hom[ f ] x* y'
    f* = has-lift f y' .lifting

    module f* = is-cartesian (has-lift f y' .cartesian)

    i* : Hom[ id ] x' x*
    i* = f*.universal' (idr f) f'

    module i*-inv = is-invertible[_] (vert-inv i*)

    i*⁻¹ : Hom[ id ] x* x'
    i*⁻¹ = i*-inv.inv'

    factors : f' ∘' i*⁻¹ ≡[ idr f ] f*
    factors = to-pathp⁻ $
      f' ∘' i*⁻¹               ≡⟨ shiftr _ (pushl' _ (symP $ f*.commutesp (idr f) f') {q = ap (f ∘_) (sym (idl _))}) ⟩
      hom[] (f* ∘' i* ∘' i*⁻¹) ≡⟨ weave _ (elimr (idl id)) _ (elimr' _ i*-inv.invl') ⟩
      hom[] f* ∎

    f-cart : is-cartesian f f'
    f-cart = cartesian-vertical-retraction-stable
      (has-lift f y' .cartesian)
      (inverses[]→from-has-section[] i*-inv.inverses')
      factors
```

As a corollary, we get that all discrete fibrations are right fibrations.
Intuitively, this is true, as sets are 0-groupoids.

```agda
discrete→right-fibration
  : is-discrete-cartesian-fibration ℰ
  → Right-fibration
discrete→right-fibration dfib =
  vertical-invertible+fibration→right-fibration
    (discrete→cartesian ℰ dfib)
    (is-discrete-cartesian-fibration.all-invertible↓ dfib)
```

## Fibred functors and right fibrations

As every map in a right fibration is cartesian, every [[displayed functor]]
into a right fibration is automatically fibred.

```agda
functor+right-fibration→fibred
  : ∀ {o₂ ℓ₂ o₂' ℓ₂'}
  → {𝒟 : Precategory o₂ ℓ₂}
  → {ℱ : Displayed 𝒟 o₂' ℓ₂'}
  → {F : Functor 𝒟 ℬ}
  → Right-fibration
  → (F' : Displayed-functor ℱ ℰ F)
  → Fibred-functor ℱ ℰ F
functor+right-fibration→fibred rfib F' .Fibred-functor.disp =
  F'
functor+right-fibration→fibred rfib F' .Fibred-functor.F-cartesian f' _ =
  Right-fibration.cartesian rfib (F₁' f')
  where open Displayed-functor F'
```

Specifically, this implies that all displayed functors into a discrete
fibrations are fibred. This means that we can drop the fibred condition
when working with functors between discrete fibrations.

```agda
functor+discrete→fibred
  : ∀ {o₂ ℓ₂ o₂' ℓ₂'}
  → {𝒟 : Precategory o₂ ℓ₂}
  → {ℱ : Displayed 𝒟 o₂' ℓ₂'}
  → {F : Functor 𝒟 ℬ}
  → is-discrete-cartesian-fibration ℰ
  → (F' : Displayed-functor ℱ ℰ F)
  → Fibred-functor ℱ ℰ F
functor+discrete→fibred disc F' =
  functor+right-fibration→fibred (discrete→right-fibration disc) F'
```
