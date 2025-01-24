<!--
```agda
open import Cat.Displayed.Cartesian.Discrete
open import Cat.Instances.Shape.Terminal
open import Cat.Displayed.Bifibration
open import Cat.Displayed.Cocartesian
open import Cat.Displayed.Cartesian
open import Cat.Functor.Equivalence
open import Cat.Displayed.Fibre
open import Cat.Displayed.Total
open import Cat.Displayed.Base
open import Cat.Prelude
```
-->

```agda
module Cat.Displayed.Instances.Identity
  {o ℓ} (B : Precategory o ℓ)
  where

open Precategory B
open Displayed
open Functor
open Total-hom
```

## The identity bifibration

Let $\cB$ be a precategory. We can define a [[displayed category]]
$\mathrm{Id}(\cB)$ over $B$ whose [[total category]] is isomorphic to
$B$, called the **identity bifibration**.

```agda
IdD : Displayed B lzero lzero
IdD .Ob[_] _ = ⊤
IdD .Hom[_] _ _ _ = ⊤
IdD .Hom[_]-set _ _ _ = hlevel 2
IdD .id' = tt
IdD ._∘'_ _ _ = tt
IdD .idr' _ = refl
IdD .idl' _ = refl
IdD .assoc' _ _ _ = refl
```

This fibration is obviously a discrete fibration; in fact, it's about as
discrete as you can get!

```agda
IdD-discrete-fibration : is-discrete-cartesian-fibration IdD
IdD-discrete-fibration .is-discrete-cartesian-fibration.fibre-set _ = hlevel 2
IdD-discrete-fibration .is-discrete-cartesian-fibration.cart-lift _ _ = hlevel 0
```

Every morphism in the identity fibration is trivially cartesian and
cocartesian.

```agda
idd-is-cartesian
  : ∀ {x y} {f : Hom x y}
  → is-cartesian IdD f tt
idd-is-cartesian .is-cartesian.universal _ _ = tt
idd-is-cartesian .is-cartesian.commutes _ _ = refl
idd-is-cartesian .is-cartesian.unique _ _ = refl

idd-is-cocartesian
  : ∀ {x y} {f : Hom x y}
  → is-cocartesian IdD f (tt)
idd-is-cocartesian .is-cocartesian.universal _ _ = tt
idd-is-cocartesian .is-cocartesian.commutes _ _ = refl
idd-is-cocartesian .is-cocartesian.unique _ _ = refl
```

This implies that it's trivially a bifibration. We omit the proofs, because they
are totally uninteresting.

```agda
IdD-fibration : Cartesian-fibration IdD
IdD-opfibration : Cocartesian-fibration IdD
IdD-bifibration : is-bifibration IdD
```
<!--
```agda
IdD-fibration .Cartesian-fibration.has-lift f y' .Cartesian-lift.x' = tt
IdD-fibration .Cartesian-fibration.has-lift f y' .Cartesian-lift.lifting = tt
IdD-fibration .Cartesian-fibration.has-lift f y' .Cartesian-lift.cartesian =
  idd-is-cartesian

IdD-opfibration .Cocartesian-fibration.has-lift f x' .Cocartesian-lift.y' = tt
IdD-opfibration .Cocartesian-fibration.has-lift f x' .Cocartesian-lift.lifting = tt
IdD-opfibration .Cocartesian-fibration.has-lift f x' .Cocartesian-lift.cocartesian =
  idd-is-cocartesian

IdD-bifibration .is-bifibration.fibration = IdD-fibration
IdD-bifibration .is-bifibration.opfibration = IdD-opfibration
```
-->

## Fibre categories

The [[fibre categories]] of the identity bifibration are isomorphic to
the [[terminal category]].

```agda
IdDFib : ∀ x → Functor ⊤Cat (Fibre IdD x)
IdDFib x .F₀ _ = tt
IdDFib x .F₁ _ = tt
IdDFib x .F-id = refl
IdDFib x .F-∘ _ _ = refl

IdD-is-iso : ∀ x → is-precat-iso (IdDFib x)
IdD-is-iso x .is-precat-iso.has-is-ff = id-equiv
IdD-is-iso x .is-precat-iso.has-is-iso = id-equiv
```

## Total category

The total category of the identity bifibration is isomorphic to $\cB$
itself.

```agda
IdDTotal : Functor B (∫ IdD)
IdDTotal .F₀ x = x , tt
IdDTotal .F₁ f = total-hom f (tt)
IdDTotal .F-id = total-hom-path _ refl refl
IdDTotal .F-∘ _ _ = total-hom-path _ refl refl

IdDTotal-is-iso : is-precat-iso IdDTotal
IdDTotal-is-iso .is-precat-iso.has-is-ff =
  is-iso→is-equiv (iso hom (λ _ → total-hom-path _ refl refl) (λ _ → refl))
IdDTotal-is-iso .is-precat-iso.has-is-iso =
  is-iso→is-equiv (iso fst (λ _ → refl) (λ _ → refl))
```

<!--
  [TODO: Reed M, 19/04/2023] Show that this is a left/right unit to composition
  of displayed categories once we have equivalence/isomorphism of displayed categories.
-->
