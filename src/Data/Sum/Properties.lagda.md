<!--
```agda
open import 1Lab.Function.Embedding
open import 1Lab.Reflection.HLevel
open import 1Lab.HLevel.Universe
open import 1Lab.HLevel.Closure
open import 1Lab.HLevel
open import 1Lab.Equiv
open import 1Lab.Path
open import 1Lab.Type

open import Data.List.Base
open import Data.Dec.Base
open import Data.Nat.Base
open import Data.Sum.Base

open import Meta.Invariant
```
-->

```agda
module Data.Sum.Properties where
```

As warmup, we have that both constructors are embeddings:

<!--
```agda
private variable
  a b c d : Level
  A B C D : Type a
```
-->

```agda
inl-inj : {x y : A} → inl {B = B} x ≡ inl y → x ≡ y
inl-inj {A = A} {x = x} = ap f module inl-inj where
  f : A ⊎ B → A
  f (inl x) = x
  f (inr _) = x

inr-inj : {A : Type b} {x y : B} → inr {A = A} x ≡ inr y → x ≡ y
inr-inj {B = B} {x = x} path = ap f path where
  f : A ⊎ B → B
  f (inl _) = x
  f (inr x) = x

inl≠inr : {A : Type a} {B : Type b} {x : A} {y : B} → ¬ inl x ≡ inr y
inl≠inr path = subst (λ { (inl x) → ⊤ ; (inr x) → ⊥ }) path tt

inr≠inl : {A : Type a} {B : Type b} {x : A} {y : B} → ¬ inr x ≡ inl y
inr≠inl path = inl≠inr (sym path)
```

## Closure under h-levels

If $A$ and $B$ are $n$-types, for $n \ge 2$, then so is their coproduct.
The way we prove this is by characterising the entire path space of `A ⊎
B` in terms of the path spaces for `A` and `B`, using a recursive
definition:

```agda
module ⊎Path where
  Code : A ⊎ B → A ⊎ B → Type (level-of A ⊔ level-of B)
  Code {B = B} (inl x) (inl y) = Lift (level-of B) (x ≡ y)
  Code (inl x) (inr y)         = Lift _ ⊥
  Code (inr x) (inl y)         = Lift _ ⊥
  Code {A = A} (inr x) (inr y) = Lift (level-of A) (x ≡ y)
```

Given a `Code`{.Agda} for a path in `A ⊎ B`, we can turn it into a
legitimate path. Agda automatically lets us ignore the cases where
the `Code`{.Agda} computes to `the empty type`{.Agda ident=⊥}.

```agda
  decode : {x y : A ⊎ B} → Code x y → x ≡ y
  decode {x = inl x} {y = inl x₁} code = ap inl (lower code)
  decode {x = inr x} {y = inr x₁} code = ap inr (lower code)
```

In the inverse direction, we have a procedure for turning paths into
codes:

```agda
  encode : {x y : A ⊎ B} → x ≡ y → Code x y
  encode {x = inl x} {y = inl y} path = lift (inl-inj path)
  encode {x = inl x} {y = inr y} path = absurd (inl≠inr path)
  encode {x = inr x} {y = inl y} path = absurd (inr≠inl path)
  encode {x = inr x} {y = inr y} path = lift (inr-inj path)
```

Now we must establish that `encode`{.Agda} and `decode`{.Agda} are
inverses. In the one direction, we can use path induction:

```agda
  decode-encode : {x y : A ⊎ B} (p : x ≡ y) → decode (encode p) ≡ p
  decode-encode = J (λ _ p → decode (encode p) ≡ p) d-e-refl where
    d-e-refl : {x : A ⊎ B} → decode (encode (λ i → x)) ≡ (λ i → x)
    d-e-refl {x = inl x} = refl
    d-e-refl {x = inr x} = refl
```

In the other direction, the proof is by case analysis, and everything
computes wonderfully to make the right-hand sides fillable by
`refl`{.Agda}:

```agda
  encode-decode : {x y : A ⊎ B} (p : Code x y) → encode (decode p) ≡ p
  encode-decode {x = inl x} {y = inl y} p = refl
  encode-decode {x = inr x} {y = inr y} p = refl
```

Thus, we have an equivalence between _codes for_ paths in `A ⊎ B` and
_actual_ paths `A ⊎ B`. Since `Code`{.Agda} has a nice computational
structure, we can establish its h-level by induction:

```agda
  Code≃Path : {x y : A ⊎ B} → (x ≡ y) ≃ Code x y
  Code≃Path = Iso→Equiv (encode , iso decode encode-decode decode-encode)
```

```agda
open ⊎Path

Code-is-hlevel : {x y : A ⊎ B} {n : Nat}
               → is-hlevel A (2 + n)
               → is-hlevel B (2 + n)
               → is-hlevel (Code x y) (suc n)
Code-is-hlevel {x = inl x} {inl y} {n} ahl bhl =
  Lift-is-hlevel (suc n) (ahl x y)
Code-is-hlevel {x = inr x} {inr y} {n} ahl bhl =
  Lift-is-hlevel (suc n) (bhl x y)
```

In the two cases where `x` and `y` match, we can use the fact that `Lift
preserves h-levels`{.Agda ident=Lift-is-hlevel} and the assumption that
`A` (or `B`) have the given h-level.

```agda
Code-is-hlevel {x = inl x} {inr y} {n} ahl bhl =
  Lift-is-hlevel (suc n) (is-prop→is-hlevel-suc λ x → absurd x)
Code-is-hlevel {x = inr x} {inl y} {n} ahl bhl =
  Lift-is-hlevel (suc n) (is-prop→is-hlevel-suc λ x → absurd x)
```

In the mismatched cases, we use the fact that `propositions have any
successor h-level`{.Agda ident=is-prop→is-hlevel-suc} to prove that `⊥` is
also at the same h-level as `A` and `B`. Thus, we have:

```agda
⊎-is-hlevel : (n : Nat)
            → is-hlevel A (2 + n)
            → is-hlevel B (2 + n)
            → is-hlevel (A ⊎ B) (2 + n)
⊎-is-hlevel n ahl bhl x y =
  Equiv→is-hlevel (1 + n) Code≃Path (Code-is-hlevel ahl bhl)

instance
  H-Level-⊎ : ∀ {n} ⦃ _ : 2 ≤ n ⦄ ⦃ _ : H-Level A n ⦄ ⦃ _ : H-Level B n ⦄ → H-Level (A ⊎ B) n
  H-Level-⊎ {n = suc (suc n)} ⦃ s≤s (s≤s p) ⦄ = hlevel-instance $
    ⊎-is-hlevel _ (hlevel (2 + n)) (hlevel (2 + n))
```

<!--
```agda
inl-is-embedding : is-embedding (inl {A = A} {B})
inl-is-embedding = cancellable→embedding' inl-inj λ p → decode-encode p

inr-is-embedding : is-embedding (inr {A = A} {B})
inr-is-embedding = cancellable→embedding' inr-inj λ p → decode-encode p

module _ {ℓ} {A : n-Type ℓ 2} where
  _ : is-hlevel (∣ A ∣ ⊎ ∣ A ∣) 5
  _ = hlevel 5
```
-->

Note that, in general, being a [[proposition]] and [[contractible]]
are not preserved under coproducts. Consider the case where `(A, a)` and
`(B, b)` are both contractible (this generalises to propositions): Then
their coproduct has two distinct points, `inl a` and `inr b`. However,
the coproduct of _disjoint_ propositions is a proposition:

```agda
disjoint-⊎-is-prop
  : is-prop A → is-prop B → ¬ (A × B)
  → is-prop (A ⊎ B)
disjoint-⊎-is-prop Ap Bp notab (inl x) (inl y) = ap inl (Ap x y)
disjoint-⊎-is-prop Ap Bp notab (inl x) (inr y) = absurd (notab (x , y))
disjoint-⊎-is-prop Ap Bp notab (inr x) (inl y) = absurd (notab (y , x))
disjoint-⊎-is-prop Ap Bp notab (inr x) (inr y) = ap inr (Bp x y)
```

## Discreteness

If `A` and `B` are [[discrete]], then so is `A ⊎ B`.

```agda
instance
  Discrete-⊎ : ⦃ _ : Discrete A ⦄ ⦃ _ : Discrete B ⦄ → Discrete (A ⊎ B)
  Discrete-⊎ {x = inl x} {inl y} = invmap (ap inl) inl-inj (x ≡? y)
  Discrete-⊎ {x = inl x} {inr y} = no inl≠inr
  Discrete-⊎ {x = inr x} {inl y} = no inr≠inl
  Discrete-⊎ {x = inr x} {inr y} = invmap (ap inr) inr-inj (x ≡? y)
```
