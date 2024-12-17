```agda
module 1Lab.Type where
```

# Universes {defines="universe"}

A **universe** is a type whose inhabitants are types. In Agda, there is
a family of universes, which, by default, is called `Set`. Rather
recently, Agda gained a flag to make `Set` not act like a keyword, and
allow renaming it in an import declaration from the `Agda.Primitive`
module.

```agda
open import Prim.Type hiding (Prop) public
```

`Type`{.Agda} is a type itself, so it's a natural question to ask: does
it belong to a universe? The answer is _yes_. However, Type can not
belong to itself, or we could reproduce [[Russell's paradox]].

To prevent this, the universes are parametrised by a _`Level`{.Agda}_,
where the collection of all `ℓ`-sized types is `Type (lsuc ℓ)`:

```agda
_ : (ℓ : Level) → Type (lsuc ℓ)
_ = λ ℓ → Type ℓ

level-of : {ℓ : Level} → Type ℓ → Level
level-of {ℓ} _ = ℓ
```

## Built-in types

We re-export the following very important types:

```agda
open import Prim.Data.Sigma public
open import Prim.Data.Bool public
open import Prim.Data.Nat hiding (_<_) public
```

Additionally, we define the empty type:

```agda
data ⊥ : Type where

absurd : ∀ {ℓ} {A : Type ℓ} → .⊥ → A
absurd ()

¬_ : ∀ {ℓ} → Type ℓ → Type ℓ
¬ A = A → ⊥
infix 6 ¬_
```

:::{.definition #product-type}
The non-dependent product type `_×_`{.Agda} can be defined in terms of
the dependent sum type:
:::

```agda
_×_ : ∀ {a b} → Type a → Type b → Type _
A × B = Σ[ _ ∈ A ] B

infixr 5 _×_
```

## Lifting

There is a function which lifts a type to a higher universe:

```agda
record Lift {a} ℓ (A : Type a) : Type (a ⊔ ℓ) where
  constructor lift
  field
    lower : A
```

<!--
```agda
open Lift public

instance
  Lift-instance : ∀ {ℓ ℓ'} {A : Type ℓ} → ⦃ A ⦄ → Lift ℓ' A
  Lift-instance ⦃ x ⦄ = lift x
```
-->

## Function composition

Since the following definitions are fundamental, they deserve a place in
this module:

```agda
_∘_ : ∀ {ℓ₁ ℓ₂ ℓ₃} {A : Type ℓ₁} {B : A → Type ℓ₂} {C : (x : A) → B x → Type ℓ₃}
    → (∀ {x} → (y : B x) → C x y)
    → (f : ∀ x → B x)
    → ∀ x → C x (f x)
f ∘ g = λ z → f (g z)

infixr 40 _∘_

id : ∀ {ℓ} {A : Type ℓ} → A → A
id x = x
{-# INLINE id #-}

infixr -1 _$_ _$ᵢ_ _$ₛ_

_$_ : ∀ {ℓ₁ ℓ₂} {A : Type ℓ₁} {B : A → Type ℓ₂} → ((x : A) → B x) → ((x : A) → B x)
f $ x = f x
{-# INLINE _$_ #-}

_$ᵢ_ : ∀ {ℓ₁ ℓ₂} {A : Type ℓ₁} {B : .A → Type ℓ₂} → (.(x : A) → B x) → (.(x : A) → B x)
f $ᵢ x = f x
{-# INLINE _$ᵢ_ #-}

_$ₛ_ : ∀ {ℓ₁ ℓ₂} {A : Type ℓ₁} {B : A → SSet ℓ₂} → ((x : A) → B x) → ((x : A) → B x)
f $ₛ x = f x
{-# INLINE _$ₛ_ #-}
```

<!--
```
open import Prim.Literals public

auto : ∀ {ℓ} {A : Type ℓ} → ⦃ A ⦄ → A
auto ⦃ a ⦄ = a

case_of_ : ∀ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'} → A → (A → B) → B
case x of f = f x

case_return_of_ : ∀ {ℓ ℓ'} {A : Type ℓ} (x : A) (B : A → Type ℓ') (f : (x : A) → B x) → B x
case x return P of f = f x

{-# INLINE case_of_        #-}
{-# INLINE case_return_of_ #-}

instance
  Number-Lift : ∀ {ℓ ℓ'} {A : Type ℓ} → ⦃ Number A ⦄ → Number (Lift ℓ' A)
  Number-Lift {ℓ' = ℓ'} ⦃ a ⦄ .Number.Constraint n = Lift ℓ' (a .Number.Constraint n)
  Number-Lift ⦃ a ⦄ .Number.fromNat n ⦃ lift c ⦄ = lift (a .Number.fromNat n ⦃ c ⦄)

infixr -1 primForce
primitive
  primForce : ∀ {a b} {A : Type a} {B : A → Type b} (x : A) (f : ∀ x → B x) → B x

syntax primForce x f = f $! x
```
-->
