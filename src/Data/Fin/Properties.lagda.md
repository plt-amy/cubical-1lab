<!--
```agda
open import 1Lab.Prelude

open import Data.Maybe.Properties
open import Data.Maybe.Base
open import Data.Fin.Base
open import Data.Nat.Base using (s≤s)
open import Data.Dec
open import Data.Irr
open import Data.Sum

open import Meta.Invariant

import Data.Nat.Order as Nat
import Data.Nat.Base as Nat
```
-->

```agda
module Data.Fin.Properties where
```

# Finite sets - properties

## Ordering

As noted in [`Data.Fin.Base`], we've set up the ordering on `Fin` so that
we can re-use all the proofs about the ordering on `Nat`{.Agda}.

[`Data.Fin.Base`]: Data.Fin.Base.html

However, there are still quite a few interesting things one can say about
`skip`{.Agda} and `squish`{.Agda}. In particular, we can prove the
*simplicial identities*, which characterize the interactions between
these two functions.

These lemmas might seem somewhat arbitrary and complicated, which
is true! However, they are enough to describe all the possible
interactions of `skip`{.Agda} and `squish`{.Agda}, which in turn
are the building blocks for _every_ monotone function between
`Fin`{.Agda}, so it's not that surprising that they would be a bit
of a mess!

<!-- [TODO: Reed M, 23/02/2022] Link to Simplicial Sets Stuff -->

```agda
skip-comm : ∀ {n} (i j : Fin (suc n)) → i ≤ j
          → ∀ x → skip (weaken i) (skip j x) ≡ skip (fsuc j) (skip i x)
skip-comm i j le x with fin-view i | fin-view j | le | fin-view x
... | zero  | zero  | _      | _       = refl
... | zero  | suc _ | _      | _       = refl
... | suc i | suc j | le     | zero    = refl
... | suc i | suc j | s≤s le | (suc x) = ap fsuc (skip-comm i j le x)

drop-comm : ∀ {n} (i j : Fin n) → i ≤ j
          → ∀ x → squish j (squish (weaken i) x) ≡ squish i (squish (fsuc j) x)
drop-comm i j le x with fin-view i | fin-view j | le | fin-view x
... | zero  | zero  | le     | zero  = refl
... | zero  | zero  | le     | suc x = refl
... | zero  | suc j | le     | zero  = refl
... | zero  | suc j | le     | suc x = refl
... | suc i | suc j | le     | zero  = refl
... | suc i | suc j | s≤s le | suc x = ap fsuc (drop-comm i j le x)

squish-skip-comm : ∀ {n} (i : Fin (suc n)) (j : Fin n) → i < fsuc j
                 → ∀ x → squish (fsuc j) (skip (weaken i) x) ≡ skip i (squish j x)
squish-skip-comm i j le x with fin-view i | fin-view j | le | fin-view x
... | zero | zero  | s≤s p | zero  = refl
... | zero | zero  | s≤s p | suc _ = refl
... | zero | suc _ | s≤s p | zero  = refl
... | zero | suc _ | s≤s p | suc _ = refl
... | suc i | (suc j) | (Nat.s≤s p) | zero = refl
... | suc i | (suc j) | (Nat.s≤s p) | (suc x) =
  ap fsuc (squish-skip-comm i j p x)

squish-skip : ∀ {n} (i j : Fin n) → i ≡ j
            → ∀ x → squish j (skip (weaken j) x) ≡ x
squish-skip i j p x with fin-view i | fin-view j | fin-view x
... | zero    | zero    | x       = refl
... | zero    | (suc j) | x       = absurd (fzero≠fsuc p)
... | (suc i) | zero    | x       = refl
... | (suc i) | (suc j) | zero    = refl
... | (suc i) | (suc j) | (suc x) =
  ap fsuc (squish-skip i j (fsuc-inj p) x)

squish-skip-fsuc : ∀ {n} (i : Fin (suc n)) (j : Fin n) → i ≡ fsuc j
                 → ∀ x → squish j (skip i x) ≡ x
squish-skip-fsuc i j p x with fin-view i | fin-view j | fin-view x
... | zero | zero | x = refl
... | zero | suc j | x = absurd (fzero≠fsuc p)
... | suc i | suc j | zero  = refl
... | suc i | suc j | suc x = ap fsuc (squish-skip-fsuc i j (fsuc-inj p) x)
... | suc i | zero | x with fin-view i | x
... | zero | zero = refl
... | zero | suc x = refl
... | suc i | zero = refl
... | suc i | suc x = absurd (Nat.zero≠suc λ i → Nat.pred (p (~ i) .lower))

Fin-suc : ∀ {n} → Fin (suc n) ≃ Maybe (Fin n)
Fin-suc = Iso→Equiv (to , iso from ir il) where
  to : ∀ {n} → Fin (suc n) → Maybe (Fin n)
  to i with fin-view i
  ... | suc i = just i
  ... | zero  = nothing

  from : ∀ {n} → Maybe (Fin n) → Fin (suc n)
  from (just x) = fsuc x
  from nothing  = fzero

  ir : is-right-inverse from to
  ir nothing = refl
  ir (just x) = refl

  il : is-left-inverse from to
  il i with fin-view i
  ... | suc i = refl
  ... | zero  = refl

Fin-peel : ∀ {l k} → Fin (suc l) ≃ Fin (suc k) → Fin l ≃ Fin k
Fin-peel {l} {k} sl≃sk = Maybe-injective (Equiv.inverse Fin-suc ∙e sl≃sk ∙e Fin-suc)

Fin-injective : ∀ {l k} → Fin l ≃ Fin k → l ≡ k
Fin-injective {zero} {zero} l≃k = refl
Fin-injective {zero} {suc k} l≃k with equiv→inverse (l≃k .snd) fzero
... | ()
Fin-injective {suc l} {zero} l≃k with l≃k .fst fzero
... | ()
Fin-injective {suc l} {suc k} sl≃sk = ap suc $ Fin-injective (Fin-peel sl≃sk)

avoid-injective
  : ∀ {n} (i : Fin (suc n)) {j k : Fin (suc n)} {i≠j : i ≠ j} {i≠k : i ≠ k}
  → avoid i j i≠j ≡ avoid i k i≠k → j ≡ k
avoid-injective i {j} {k} {i≠j} {i≠k} p with fin-view i | fin-view j | fin-view k
... | zero | zero | _ = absurd (i≠j refl)
... | zero | suc j | zero = absurd (i≠k refl)
... | zero | suc j | suc k = ap fsuc p
... | suc i | zero | zero = refl
avoid-injective {suc n} _ p | suc i | zero  | suc k = absurd (fzero≠fsuc p)
avoid-injective {suc n} _ p | suc i | suc j | zero  = absurd (fsuc≠fzero p)
avoid-injective {suc n} _ p | suc i | suc j | suc k = ap fsuc (avoid-injective {n} i {j} {k} (fsuc-inj p))

skip-injective
  : ∀ {n} (i : Fin (suc n)) (j k : Fin n)
  → skip i j ≡ skip i k → j ≡ k
skip-injective i j k p with fin-view i | fin-view j | fin-view k
... | zero  | j     | k     = fsuc-inj p
... | suc i | zero  | zero  = refl
... | suc i | zero  | suc k = absurd (fzero≠fsuc p)
... | suc i | suc j | zero  = absurd (fsuc≠fzero p)
... | suc i | suc j | suc k = ap fsuc (skip-injective i j k (fsuc-inj p))

skip-skips
  : ∀ {n} (i : Fin (suc n)) (j : Fin n)
  → skip i j ≠ i
skip-skips i j p with fin-view i | fin-view j
... | zero  | j     = fsuc≠fzero p
... | suc i | zero  = fzero≠fsuc p
... | suc i | suc j = skip-skips i j (fsuc-inj p)

avoid-skip
  : ∀ {n} (i : Fin (suc n)) (j : Fin n) {neq : i ≠ skip i j}
  → avoid i (skip i j) neq ≡ j
avoid-skip i j with fin-view i | fin-view j
... | zero  | zero  = refl
... | zero  | suc j = refl
... | suc i | zero  = refl
... | suc i | suc j = ap fsuc (avoid-skip i j)

skip-avoid
  : ∀ {n} (i : Fin (suc n)) (j : Fin (suc n)) {i≠j : i ≠ j}
  → skip i (avoid i j i≠j) ≡ j
skip-avoid i j {i≠j} with fin-view i | fin-view j
... | zero | zero = absurd (i≠j refl)
skip-avoid {suc n} _ _ | zero  | suc j = refl
skip-avoid {suc n} _ _ | suc i | zero  = refl
skip-avoid {suc n} _ _ | suc i | suc j = ap fsuc (skip-avoid i j)
```

## Iterated products and sums {defines="iterated-products"}

We can break down $\Pi$-types and $\Sigma$-types over finite sets
as iterated products and sums, respectively.

```agda
Fin-suc-Π
  : ∀ {ℓ} {n} {A : Fin (suc n) → Type ℓ}
  → (∀ x → A x) ≃ (A fzero × (∀ x → A (fsuc x)))
Fin-suc-Π = Iso→Equiv λ where
  .fst f → f fzero , (λ x → f (fsuc x))

  .snd .is-iso.inv (z , s) → fin-cons z s

  .snd .is-iso.rinv x → refl

  .snd .is-iso.linv k i fzero               → k (fin zero ⦃ forget auto ⦄)
  .snd .is-iso.linv k i (fin (suc n) ⦃ b ⦄) → k (fin (suc n) ⦃ b ⦄)

Fin-suc-Σ
  : ∀ {ℓ} {n} {A : Fin (suc n) → Type ℓ}
  → Σ (Fin (suc n)) A ≃ (A fzero ⊎ Σ (Fin n) (A ∘ fsuc))
Fin-suc-Σ {A = A} = Iso→Equiv (to , iso from ir il) where
  to : ∫ₚ A → A fzero ⊎ ∫ₚ (A ∘ fsuc)
  to (i , a) with fin-view i
  ... | zero  = inl a
  ... | suc x = inr (x , a)

  from : A fzero ⊎ ∫ₚ (A ∘ fsuc) → ∫ₚ A
  from (inl x)       = fzero , x
  from (inr (x , a)) = fsuc x , a

  ir : is-right-inverse from to
  ir (inl x) = refl
  ir (inr x) = refl

  il : is-left-inverse from to
  il (i , a) with fin-view i
  ... | zero  = refl
  ... | suc _ = refl
```

## Finite choice {defines="finite-choice"}

An important fact about the [[(standard) finite sets|standard finite
sets]] in constructive mathematics is that they _always_ support choice,
which we phrase below as a "search" operator: if $M$ is any `Monoidal`{.Agda}
functor on types, then it commutes with products. Since $\Pi$-types
over $[n]$ are $n$-ary [[iterated products]], we have that $M$ commutes
with $\Pi$.

```agda
Fin-Monoidal
  : ∀ {ℓ} n {A : Fin n → Type ℓ} {M}
      (let module M = Effect M)
  → ⦃ Monoidal M ⦄
  → (∀ x → M.₀ (A x)) → M.₀ (∀ x → A x)
Fin-Monoidal zero _ = invmap (λ _ ()) _ munit
Fin-Monoidal (suc n) k =
  Fin-suc-Π e⁻¹ <≃> (k 0 <,> Fin-Monoidal n (k ∘ fsuc))
```

<!--
```agda
_ = Idiom
```
-->

In particular, instantiating $M$ with the [[propositional truncation]]
(which is an `Idiom`{.Agda} and hence `Monoidal`{.Agda}), we get a
version of the [[axiom of choice]] for finite sets.

```agda
finite-choice
  : ∀ {ℓ} n {A : Fin n → Type ℓ}
  → (∀ x → ∥ A x ∥) → ∥ (∀ x → A x) ∥
finite-choice n = Fin-Monoidal n
```

An immediate consequence is that surjections into a finite set (thus,
_between_ finite sets) [[merely]] split:

```agda
finite-surjection-split
  : ∀ {ℓ} {n} {B : Type ℓ}
  → (f : B → Fin n) → is-surjective f
  → ∥ (∀ x → fibre f x) ∥
finite-surjection-split f = finite-choice _
```

Dually, we have that any `Alternative`{.Agda} functor $M$ commutes with
$\Sigma$-types on finite sets, since those are iterated sums.

```agda
Fin-Alternative
  : ∀ {ℓ} n {A : Fin n → Type ℓ} {M}
      (let module M = Effect M)
  → ⦃ Alternative M ⦄
  → (∀ x → M.₀ (A x)) → M.₀ (Σ (Fin n) A)
Fin-Alternative zero _ = invmap (λ ()) (λ ()) empty
Fin-Alternative (suc n) k =
  Fin-suc-Σ e⁻¹ <≃> (k 0 <+> Fin-Alternative n (k ∘ fsuc))
```

::: {.definition #omniscience-of-finite-sets}
As a consequence, instantiating $M$ with `Dec`{.Agda}, we get that
finite sets are **exhaustible** and **omniscient**, which means that
any family of decidable types indexed by a finite sets yields decidable
$\Pi$-types and $\Sigma$-types, respectively.
:::

```agda
instance
  Dec-Fin-∀
    : ∀ {n ℓ} {A : Fin n → Type ℓ}
    → ⦃ ∀ {x} → Dec (A x) ⦄ → Dec (∀ x → A x)
  Dec-Fin-∀ {n} ⦃ d ⦄ = Fin-Monoidal n (λ _ → d)

  Dec-Fin-Σ
    : ∀ {n ℓ} {A : Fin n → Type ℓ}
    → ⦃ ∀ {x} → Dec (A x) ⦄ → Dec (Σ (Fin n) A)
  Dec-Fin-Σ {n} ⦃ d ⦄ = Fin-Alternative n λ _ → d
```

```agda
Fin-omniscience
  : ∀ {n ℓ} (P : Fin n → Type ℓ) ⦃ _ : ∀ {x} → Dec (P x) ⦄
  → (Σ[ j ∈ Fin n ] P j × ∀ k → P k → j ≤ k) ⊎ (∀ x → ¬ P x)
Fin-omniscience {zero} P = inr λ ()
Fin-omniscience {suc n} P with holds? (P 0)
... | yes here = inl (0 , here , λ _ _ → 0≤x)
... | no ¬here with Fin-omniscience (P ∘ fsuc)
... | inl (ix , pix , least) = inl (fsuc ix , pix , fin-cons (λ here → absurd (¬here here)) λ i pi → Nat.s≤s (least i pi))
... | inr nowhere = inr (fin-cons ¬here nowhere)
```

<!--
```agda
Fin-omniscience-neg
  : ∀ {n ℓ} (P : Fin n → Type ℓ) ⦃ _ : ∀ {x} → Dec (P x) ⦄
  → (∀ x → P x) ⊎ (Σ[ j ∈ Fin n ] ¬ P j × ∀ k → ¬ P k → j ≤ k)
Fin-omniscience-neg P with Fin-omniscience (¬_ ∘ P)
... | inr p = inl λ i → dec→dne (p i)
... | inl (j , ¬pj , least) = inr (j , ¬pj , least)

Fin-find
  : ∀ {n ℓ} {P : Fin n → Type ℓ} ⦃ _ : ∀ {x} → Dec (P x) ⦄
  → ¬ (∀ x → P x)
  → Σ[ x ∈ Fin n ] ¬ P x × ∀ y → ¬ P y → x ≤ y
Fin-find {P = P} ¬p with Fin-omniscience-neg P
... | inl p = absurd (¬p p)
... | inr p = p
```
-->

## Injections and surjections

The standard finite sets are **Dedekind-finite**, which means that every
injection $[n] \mono [n]$ is a bijection. We prove this by a straightforward
but annoying induction on $n$.

```agda
Fin-injection→equiv
  : ∀ {n} (f : Fin n → Fin n)
  → injective f → is-equiv f
Fin-injection→equiv {zero} f inj .is-eqv ()
Fin-injection→equiv {suc n} f inj .is-eqv i with f 0 ≡? i
... | yes p = contr (0 , p) λ (j , p') → Σ-prop-path! (inj (p ∙ sym p'))
... | no ¬p = contr fib cen where
  rec = Fin-injection→equiv {n}
    (λ x → avoid (f 0) (f (fsuc x)) (Nat.zero≠suc ∘ ap lower ∘ inj))
    (λ p → fsuc-inj (inj (avoid-injective (f 0) p)))
    .is-eqv (avoid (f 0) i ¬p)

  fib : fibre f i
  fib = fsuc (rec .centre .fst) , avoid-injective (f 0) (rec .centre .snd)

  cen : ∀ x → fib ≡ x
  cen (i , p) with fin-view i
  ... | zero  = absurd (¬p p)
  ... | suc j = Σ-prop-path! (ap (fsuc ∘ fst)
      (rec .paths (j , ap₂ (avoid (f 0)) p prop!)))
```

Since [[every surjection between finite sets splits|finite choice]], any
surjection $f : [n] \epi [n]$ has an injective right inverse, which is thus
a bijection; by general properties of equivalences, this implies that $f$ is
also a bijection.

```agda
Fin-surjection→equiv
  : ∀ {n} (f : Fin n → Fin n)
  → is-surjective f → is-equiv f
Fin-surjection→equiv f surj = case finite-surjection-split f surj of λ split →
  left-inverse→equiv (snd ∘ split)
    (Fin-injection→equiv (fst ∘ split)
      (right-inverse→injective f (snd ∘ split)))
```

## Vector operations

```agda
avoid-insert
  : ∀ {n} {ℓ} {A : Type ℓ}
  → (ρ : Fin n → A)
  → (i : Fin (suc n)) (a : A)
  → (j : Fin (suc n))
  → (i≠j : i ≠ j)
  → (ρ [ i ≔ a ]) j ≡ ρ (avoid i j i≠j)
avoid-insert ρ i a j i≠j with fin-view i | fin-view j
... | zero | zero   = absurd (i≠j refl)
... | zero | suc j  = refl
avoid-insert {suc n} ρ _ a _ _   | suc i | zero = refl
avoid-insert {suc n} ρ _ a _ i≠j | suc i | suc j =
  avoid-insert (ρ ∘ fsuc) i a j (i≠j ∘ ap fsuc)

insert-lookup
  : ∀ {n} {ℓ} {A : Type ℓ}
  → (ρ : Fin n → A)
  → (i : Fin (suc n)) (a : A)
  → (ρ [ i ≔ a ]) i ≡ a
insert-lookup {n = n} ρ i a with fin-view i
... | zero = refl
insert-lookup {n = suc n} ρ _ a | suc i = insert-lookup (ρ ∘ fsuc) i a

delete-insert
  : ∀ {n} {ℓ} {A : Type ℓ}
  → (ρ : Fin n → A)
  → (i : Fin (suc n)) (a : A)
  → ∀ j → delete (ρ [ i ≔ a ]) i j ≡ ρ j
delete-insert ρ i a j with fin-view i | fin-view j
... | zero  | j       = refl
... | suc i | zero    = refl
... | suc i | (suc j) = delete-insert (ρ ∘ fsuc) i a j

insert-delete
  : ∀ {n} {ℓ} {A : Type ℓ}
  → (ρ : Fin (suc n) → A)
  → (i : Fin (suc n)) (a : A)
  → ρ i ≡ a
  → ∀ j → ((delete ρ i) [ i ≔ a ]) j ≡ ρ j
insert-delete ρ i a p j with fin-view i | fin-view j
... | zero  | zero  = sym p
... | zero  | suc j = refl
insert-delete {suc n} ρ _ a p _ | suc i | zero  = refl
insert-delete {suc n} ρ _ a p _ | suc i | suc j = insert-delete (ρ ∘ fsuc) i a p j

ℕ< : Nat → Type
ℕ< n = Σ[ k ∈ Nat ] k Nat.< n

from-ℕ< : ∀ {n} → ℕ< n → Fin n
from-ℕ< (i , p) = fin i ⦃ forget p ⦄

to-ℕ< : ∀ {n} → Fin n → ℕ< n
to-ℕ< (fin i ⦃ forget p ⦄) = i , recover p

fsuc-is-embedding : ∀ {n} → is-embedding (fsuc {n})
fsuc-is-embedding = injective→is-embedding! fsuc-inj
```
