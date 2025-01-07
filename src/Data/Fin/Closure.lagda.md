<!--
```agda
open import 1Lab.Prelude

open import Data.Maybe.Properties
open import Data.Set.Coequaliser
open import Data.Fin.Properties
open import Data.Nat.Properties
open import Data.Nat.DivMod
open import Data.Nat.Order
open import Data.Fin.Base
open import Data.Nat.Base as Nat
open import Data.Dec
open import Data.Irr
open import Data.Sum

open import Meta.Invariant

open is-iso
```
-->

```agda
module Data.Fin.Closure where
```

<!--
```agda
private variable
  ℓ : Level
  A B C : Type ℓ
  k l m n : Nat
```
-->

# Closure of finite sets {defines="closure-of-finite-sets"}

In this module, we prove that the finite sets are closed under "typal
arithmetic": The initial and terminal objects are finite (they have 1
and 0 elements respectively), products of finite sets are finite,
coproducts of finite sets are finite, and functions between finite sets
are finite. Moreover, these operations all correspond to arithmetic
operations on the natural number indices: $[n] \uplus [m] = [n + m]$,
etc.

## Zero, one, successors

The finite set $[0]$ is an initial object, and the finite set $[1]$ is a
[[terminal object]]:

```agda
Finite-zero-is-initial : Fin 0 ≃ ⊥
Finite-zero-is-initial .fst ()
Finite-zero-is-initial .snd .is-eqv ()

Finite-one-is-contr : is-contr (Fin 1)
Finite-one-is-contr .centre = fzero
Finite-one-is-contr .paths i with fin-view i
... | zero = refl
```

The successor operation on indices corresponds to taking coproducts with
the unit set.

```agda
Finite-successor : Fin (suc n) ≃ (⊤ ⊎ Fin n)
Finite-successor {n} = Fin-suc ∙e Maybe-is-sum
```

## Addition

For binary coproducts, we prove the correspondence with addition in
steps, to make the proof clearer:

<!--
```agda
module _ where private
```
-->

```agda
  Finite-coproduct : (Fin n ⊎ Fin m) ≃ Fin (n + m)
  Finite-coproduct {zero} {m}  =
    (Fin 0 ⊎ Fin m) ≃⟨ ⊎-apl Finite-zero-is-initial ⟩
    (⊥ ⊎ Fin m)     ≃⟨ ⊎-zerol ⟩
    Fin m           ≃∎
  Finite-coproduct {suc n} {m} =
    (Fin (suc n) ⊎ Fin m) ≃⟨ ⊎-apl Finite-successor ⟩
    ((⊤ ⊎ Fin n) ⊎ Fin m) ≃⟨ ⊎-assoc ⟩
    (⊤ ⊎ (Fin n ⊎ Fin m)) ≃⟨ ⊎-apr (Finite-coproduct {n} {m}) ⟩
    (⊤ ⊎ Fin (n + m))     ≃⟨ Finite-successor e⁻¹ ⟩
    Fin (suc (n + m))     ≃∎
```

<!--
```agda
Finite-coproduct : ∀ {m n} → (Fin m ⊎ Fin n) ≃ Fin (m + n)
Finite-coproduct {m} {n} = Iso→Equiv (to , iso from ir il) where
  to : Fin m ⊎ Fin n → Fin (m + n)
  to (inl x) = record
    { lower   = x .lower
    ; bounded = forget (≤-trans (to-ℕ< x .snd) (+-≤l m n))
    }
  to (inr (fin i ⦃ forget α ⦄)) =
    let
      .p : m + i Nat.< m + n
      p = ≤-trans (≤-refl' (sym (+-sucr m i))) (+-preserves-≤ m m (suc _) n ≤-refl α)
    in record
      { lower   = m + i
      ; bounded = forget p
      }

  from : Fin (m + n) → Fin m ⊎ Fin n
  from (fin i ⦃ forget b ⦄) with holds? (i Nat.< m)
  ... | yes p = inl (fin i ⦃ forget p ⦄)
  ... | no ¬p =
    let
      p' : m Nat.≤ i
      p' = ≤-peel (<-from-not-≤ _ _ ¬p)

      q : i - m Nat.≤ i
      q = monus-≤ i m

      .r : i - m Nat.< n
      r = +-reflects-≤l (suc (i - m)) n m (≤-trans (≤-refl' (+-sucr m (i - m))) (≤-trans (≤-refl' (ap suc (monus-inversel i m p'))) b))
    in inr (fin (i - m) ⦃ forget r ⦄)

  ir : is-right-inverse from to
  ir (fin i ⦃ forget b ⦄) with holds? (i Nat.< m)
  ... | yes p = fin-ap refl
  ... | no ¬p = fin-ap (monus-inversel i m (≤-peel (<-from-not-≤ _ _ ¬p)))

  il : is-left-inverse from to
  il (inl (fin i ⦃ forget b ⦄)) with holds? (i Nat.< m)
  ... | yes p = refl
  ... | no ¬p = absurd (¬p b)
  il (inr (fin i ⦃ forget b ⦄)) with holds? ((m + i) Nat.< m)
  ... | yes p = absurd (¬sucx≤x m (+-reflects-≤l (suc m) m i (≤-trans (≤-refl' (+-sucr i m ∙ ap suc (+-commutative i m))) (≤-trans p (+-≤r i m)))))
  ... | no ¬p = ap inr (fin-ap (monus-inverser i m))
```
-->

### Sums

We also have a correspondence between "coproducts" and "addition" in the
iterated case: If you have a family of finite types (represented by a
map to their cardinalities), the dependent _sum_ of that family is
equivalent to the iterated binary `sum`{.Agda} of the cardinalities:

```agda
sum : ∀ n → (Fin n → Nat) → Nat
sum zero f = zero
sum (suc n) f = f fzero + sum n (f ∘ fsuc)

Finite-sum : (B : Fin n → Nat) → Σ (Fin _) (Fin ∘ B) ≃ Fin (sum n B)
Finite-sum {zero} B .fst ()
Finite-sum {zero} B .snd .is-eqv ()
Finite-sum {suc n} B =
  Σ (Fin (suc n)) (Fin ∘ B)              ≃⟨ Fin-suc-Σ ⟩
  Fin (B 0) ⊎ Σ (Fin n) (Fin ∘ B ∘ fsuc) ≃⟨ ⊎-apr (Finite-sum (B ∘ fsuc)) ⟩
  Fin (B 0) ⊎ Fin (sum n (B ∘ fsuc))     ≃⟨ Finite-coproduct ⟩
  Fin (sum (suc n) B)                    ≃∎
```

## Multiplication

Recall (from middle school) that the product $n \times m$ is the same
thing as summing together $n$ copies of the number $m$. Correspondingly,
we can use the theorem above for general sums to establish the case of
binary products:

<!--
```agda
module _ where private
```
-->

```agda
  Finite-multiply : (Fin n × Fin m) ≃ Fin (n * m)
  Finite-multiply {n} {m} =
    (Fin n × Fin m)       ≃⟨ Finite-sum (λ _ → m) ⟩
    Fin (sum n (λ _ → m)) ≃⟨ path→equiv (ap Fin (sum≡* n m)) ⟩
    Fin (n * m)           ≃∎
    where
      sum≡* : ∀ n m → sum n (λ _ → m) ≡ n * m
      sum≡* zero m = refl
      sum≡* (suc n) m = ap (m +_) (sum≡* n m)
```

<!--
```agda
Finite-multiply : ∀ {m n} → (Fin m × Fin n) ≃ Fin (m * n)
Finite-multiply {zero} {n} = fst , record { is-eqv = λ o → absurd (Fin-absurd o) }
Finite-multiply {suc n} {zero} = ((λ (_ , x) → absurd (Fin-absurd x))) , record { is-eqv = λ o → absurd (Fin-absurd (subst Fin (*-zeror n) o)) }
Finite-multiply {m@(suc m')} {n@(suc n')} = Iso→Equiv (to , iso from ir il) where
  to : Fin m × Fin n → Fin (m * n)
  to (fin i ⦃ forget b ⦄ , fin j ⦃ forget b' ⦄) = fin (i * n + j) ⦃ forget α ⦄ where
    α : i * n + j Nat.< m * n
    α =
      let
        it : i * n + j Nat.≤ m' * n + n'
        it = +-preserves-≤ (i * n) (m' * n) j n' (*-preserves-≤r i m' n (≤-peel (recover b))) (≤-peel (recover b'))
      in s≤s (≤-trans it  (≤-refl' (+-commutative (m' * n) n')))

  from : Fin (m * n) → Fin m × Fin n
  from (fin i ⦃ forget b ⦄) with divmod q r quot rem ← divide-pos i n =
    let
      .b' : q Nat.≤ m
      b' = *-cancel-≤r n {q} {m} $
        ≤-trans (difference→≤ r (sym quot)) (≤-sucr (≤-peel b))

      .ne : q ≠ m
      ne p =
        let
          p' : m * n Nat.≤ i
          p' = difference→≤ r (sym (subst (λ e → i ≡ e * suc n' + r) p quot))
        in ¬sucx≤x _ (≤-trans b p')

    in fin q ⦃ forget (<-from-≤ ne b') ⦄ , fin r ⦃ forget rem ⦄

  ir : is-right-inverse from to
  ir (fin i ⦃ forget b ⦄) = fin-ap (sym (is-divmod i n))

  il : is-left-inverse from to
  il (fin i ⦃ forget b ⦄ , fin j ⦃ forget b' ⦄) =
    let
      p : Path (DivMod (i * n + j) n) (divide-pos (i * n + j) n) (divmod i j refl b')
      p = prop!
    in fin-ap (ap DivMod.quot p) ,ₚ fin-ap (ap DivMod.rem p)
```
-->

### Products

Similarly to the case for sums, the cardinality of a dependent *product* of
finite sets is the `product`{.Agda} of the cardinalities:

```agda
product : ∀ n → (Fin n → Nat) → Nat
product zero f = 1
product (suc n) f = f fzero * product n (f ∘ fsuc)

Finite-product : (B : Fin n → Nat) → (∀ x → Fin (B x)) ≃ Fin (product n B)
Finite-product {zero} B .fst _ = fzero
Finite-product {zero} B .snd = is-iso→is-equiv λ where
  .is-iso.inv  _ ()
  .is-iso.linv _ → funext λ ()

  .is-iso.rinv fzero                      → refl
  .is-iso.rinv (fin (suc i) ⦃ forget p ⦄) → absurd (¬suc≤0 (≤-peel p))
Finite-product {suc n} B =
  (∀ x → Fin (B x))                          ≃⟨ Fin-suc-Π ⟩
  Fin (B fzero) × (∀ x → Fin (B (fsuc x)))   ≃⟨ Σ-ap-snd (λ _ → Finite-product (B ∘ fsuc)) ⟩
  Fin (B fzero) × Fin (product n (B ∘ fsuc)) ≃⟨ Finite-multiply ⟩
  Fin (B fzero * product n (B ∘ fsuc))       ≃∎
```

## Permutations

We show that the set of permutations $[n] \simeq [n]$ is finite with
cardinality $n!$ (the `factorial`{.Agda} of $n$).

We start by showing that $([n+1] \simeq [n+1]) \simeq [n+1] \times
([n] \simeq [n])$: a permutation of $[n+1]$ is determined by what happens
to $0$ and by the remaining permutation of $[n]$.

```agda
Fin-permutations-suc
  : ∀ n → (Fin (suc n) ≃ Fin (suc n)) ≃ (Fin (suc n) × (Fin n ≃ Fin n))
Fin-permutations-suc n = to , is-iso→is-equiv is where
  to : (Fin (suc n) ≃ Fin (suc n)) → Fin (suc n) × (Fin n ≃ Fin n)
  to e .fst = e # 0
  to e .snd .fst i = avoid (e # 0) (e # (fsuc i)) λ p →
    zero≠suc (ap lower (Equiv.injective e p))
  to e .snd .snd = Fin-injection→equiv _ λ p →
    fsuc-inj (Equiv.injective e (avoid-injective (e # 0) p))

  is : is-iso to
  is .is-iso.inv (n , e) = record { fst = fun ; snd = Fin-injection→equiv _ inj  } module inv where
    fun : Fin (suc _) → Fin (suc _)
    fun i with fin-view i
    ... | zero  = n
    ... | suc x = skip n (e # x)

    inj : injective fun
    inj {i} {j} p with fin-view i | fin-view j
    ... | zero  | zero  = refl
    ... | zero  | suc y = absurd (skip-skips n _ (sym p))
    ... | suc i | zero  = absurd (skip-skips n _ p)
    ... | suc i | suc j = ap fsuc (Equiv.injective e (skip-injective n _ _ p))
  is .is-iso.rinv (n , e) = Σ-pathp refl (ext λ i → avoid-skip n (e # i))
  is .is-iso.linv e = ext p where
    p : ∀ x → inv.fun (e # 0) (to e .snd) x ≡ e .fst x
    p x with fin-view x
    ... | zero  = refl
    ... | suc i = skip-avoid (e # 0) (e # fsuc i)
```

We can now show that $([n] \simeq [n]) \simeq [n!]$ by induction.

```agda
Fin-permutations : ∀ n → (Fin n ≃ Fin n) ≃ Fin (factorial n)
Fin-permutations zero = is-contr→≃
  (contr id≃ λ _ → ext λ ()) Finite-one-is-contr
Fin-permutations (suc n) =
  Fin (suc n) ≃ Fin (suc n)       ≃⟨ Fin-permutations-suc n ⟩
  Fin (suc n) × (Fin n ≃ Fin n)   ≃⟨ Σ-ap-snd (λ _ → Fin-permutations n) ⟩
  Fin (suc n) × Fin (factorial n) ≃⟨ Finite-multiply ⟩
  Fin (factorial (suc n))         ≃∎
```

## Decidable subsets

Given a [[decidable]] predicate on $[n]$, we can compute $s$ such that
$[s]$ is equivalent to the subset of $[n]$ on which the predicate holds:
a decidable proposition is finite (it has either $0$ or $1$ elements),
so we can reuse our proof that finite sums of finite types are finite.

```agda
Dec→Fin
  : ∀ {ℓ} {A : Type ℓ} → is-prop A → Dec A
  → Σ Nat λ n → Fin n ≃ A
Dec→Fin ap (no ¬a) .fst = 0
Dec→Fin ap (no ¬a) .snd =
  is-empty→≃ (Finite-zero-is-initial .fst) ¬a
Dec→Fin ap (yes a) .fst = 1
Dec→Fin ap (yes a) .snd =
  is-contr→≃ Finite-one-is-contr (is-prop∙→is-contr ap a)

Finite-subset
  : ∀ {n} (P : Fin n → Type ℓ)
  → ⦃ ∀ {x} → H-Level (P x) 1 ⦄
  → ⦃ dec : ∀ {x} → Dec (P x) ⦄
  → Σ Nat λ s → Fin s ≃ Σ (Fin n) P
Finite-subset {n = n} P ⦃ dec = dec ⦄ =
  sum n ns , Finite-sum ns e⁻¹ ∙e Σ-ap-snd es
  where
    ns : Fin n → Nat
    ns i = Dec→Fin (hlevel 1) dec .fst
    es : (i : Fin n) → Fin (ns i) ≃ P i
    es i = Dec→Fin (hlevel 1) dec .snd
```

## Decidable quotients

As a first step towards coequalisers, we show that the [[quotient]]
of a finite set $[n]$ by a [[decidable]] [[congruence]] is finite.

```agda
Finite-quotient
  : ∀ {n ℓ} (R : Congruence (Fin n) ℓ) (open Congruence R)
  → ⦃ _ : ∀ {x y} → Dec (x ∼ y) ⦄
  → Σ Nat λ q → Fin q ≃ Fin n / _∼_
```

This amounts to counting the number of equivalence classes of $R$. We
proceed by induction on $n$; the base case is trivial.

```agda
Finite-quotient {zero} R .fst = 0
Finite-quotient {zero} R .snd .fst ()
Finite-quotient {zero} R .snd .snd .is-eqv = elim! λ ()
```

For the induction step, we restrict $R$ along the successor map $[n]
\to [n+1]$ to get a congruence $R_1$ on $[n]$ whose quotient is finite.

```agda
Finite-quotient {suc n} {ℓ} R = go where
  module R = Congruence R

  R₁ : Congruence (Fin n) ℓ
  R₁ = Congruence-pullback fsuc R
  module R₁ = Congruence R₁

  n/R₁ : Σ Nat λ q → Fin q ≃ Fin n / R₁._∼_
  n/R₁ = Finite-quotient {n} R₁
```

In order to compute the size of the quotient $[n+1]/R$, we decide whether
$0 : [n+1]$ is related by $R$ to any $i+1$ using the [[omniscience of
finite sets]].

```agda
  go
    : ⦃ Dec (Σ (Fin n) (λ i → fzero R.∼ fsuc i)) ⦄
    → Σ Nat (λ q → Fin q ≃ Fin (suc n) / R._∼_)
```

If so, $0$ lives in the same equivalence class as $i+1$ and the size of
the quotient remains unchanged.

```agda
  go ⦃ yes (i , r) ⦄ .fst = n/R₁ .fst
  go ⦃ yes (i , r) ⦄ .snd = n/R₁ .snd ∙e Iso→Equiv is where
    is : Iso (Fin n / R₁._∼_) (Fin (suc n) / R._∼_)
    is .fst = Coeq-rec (λ x → inc (fsuc x)) λ (x , y , s) → quot s
    is .snd .inv = Coeq-rec fn λ (i , j , s) → resp i j s where
      fn : Fin (suc n) → Fin n / R₁._∼_
      fn j with fin-view j
      ... | zero  = inc i
      ... | suc x = inc x

      resp : ∀ i j → i R.∼ j → fn i ≡ fn j
      resp i j s with fin-view i | fin-view j
      ... | zero  | zero  = refl
      ... | zero  | suc y = quot (R.symᶜ r R.∙ᶜ s)
      ... | suc x | zero  = quot (s R.∙ᶜ r)
      ... | suc x | suc y = quot s
    is .snd .rinv = elim! (Fin-cases (quot (R.symᶜ r)) (λ _ → refl))
    is .snd .linv = elim! λ _ → refl
```

Otherwise, $0$ creates a new equivalence class for itself.

```agda
  go ⦃ no ¬r ⦄ .fst = suc (n/R₁ .fst)
  go ⦃ no ¬r ⦄ .snd = Finite-successor ∙e ⊎-apr (n/R₁ .snd) ∙e Iso→Equiv is where
    to : Fin (suc n) → ⊤ ⊎ (Fin n / R₁._∼_)
    to i with fin-view i
    ... | zero  = inl _
    ... | suc x = inr (inc x)

    resp : ∀ i j → i R.∼ j → to i ≡ to j
    resp i j s with fin-view i | fin-view j
    ... | zero  | zero  = refl
    ... | zero  | suc y = absurd (¬r (y , s))
    ... | suc x | zero  = absurd (¬r (x , R.symᶜ s))
    ... | suc x | suc y = ap inr (quot s)

    is : Iso (⊤ ⊎ (Fin n / R₁._∼_)) (Fin (suc n) / R._∼_)
    is .fst (inl tt) = inc 0
    is .fst (inr x) = Coeq-rec (λ x → inc (fsuc x)) (λ (x , y , s) → quot s) x
    is .snd .inv = Coeq-rec to λ (x , y , r) → resp x y r
    is .snd .rinv = elim! (Fin-cases refl (λ _ → refl))
    is .snd .linv (inl tt) = refl
    is .snd .linv (inr x) = elim x where
      elim : ∀ x → is .snd .inv (is .fst (inr x)) ≡ inr x
      elim = elim! λ _ → refl
```

## Coequalisers

Given two functions $f, g : [m] \to [n]$, we can compute $q$ such that
$[q]$ is equivalent to the [[coequaliser|set coequaliser]] of $f$ and $g$.
We start by expressing the coequaliser as the quotient of $[n]$ by the
[[congruence generated by|congruence closure]] the relation $f(a) \sim
g(a)$, so that we can apply the result above. Since this relation is
clearly [[decidable]] by the [[omniscience|omniscience of finite sets]]
of $[m]$, all that remains is to show that the [[closure|congruence
closure]] of a decidable relation on a finite set is decidable.

```agda
instance
  Closure-Fin-Dec
    : ∀ {n ℓ} {R : Fin n → Fin n → Type ℓ}
    → ⦃ ∀ {x y} → Dec (R x y) ⦄
    → ∀ {x y} → Dec (Closure R x y)
```

This amounts to writing a (verified!) pathfinding algorithm: given
$x, y : [n]$, we need to decide whether there is a path between $x$
and $y$ in the undirected graph whose edges are given by $R$.

We proceed by induction on $n$; the base case is trivial, so we are
left with the inductive case $[n+1]$.
The simplest^[In terms of ease of implementation; the complexity of the
resulting algorithm is *catastrophic*.] way forward is to find a
decidable congruence $D$ that is equivalent to the closure $R^*$.

We start by defining the restriction of $R$ along the successor map
$[n] \to [n+1]$, written $R_1$, as well as the *symmetric closure* of $R$,
written $R^s$.

```agda
  Closure-Fin-Dec {suc n} {ℓ} {R} {x} {y} = R*-dec where
    open Congruence
    module R = Congruence (Closure-congruence R)

    R₁ : Fin n → Fin n → Type _
    R₁ x y = R (fsuc x) (fsuc y)
    module R₁ = Congruence (Closure-congruence R₁)

    R₁→R : ∀ {x y} → Closure R₁ x y → Closure R (fsuc x) (fsuc y)
    R₁→R = Closure-rec-congruence R₁
      (Congruence-pullback fsuc (Closure-congruence R)) inc

    Rˢ : Fin (suc n) → Fin (suc n) → Type _
    Rˢ x y = R x y ⊎ R y x

    Rˢ→R : ∀ {x y} → Rˢ x y → Closure R x y
    Rˢ→R = [ inc , R.symᶜ ∘ inc ]
```

We build $D$ by cases. $D(0, 0)$ is trivial, since the closure is reflexive.

```agda
    D' : {i j : Fin (suc n)} → Fin-view i → Fin-view j → Type ℓ
    D' zero zero = Lift _ ⊤
```

For $D(0, y+1)$, we use the omniscience of $[n]$ to search for an $x$
such that $R^s(0, x+1)$ and $R_1^*(x, y)$. Here we rely on the closure
of $R_1$ being decidable by the induction hypothesis.
The case $D(x+1, 0)$ is symmetric.

```agda
    D' zero (suc y) = Σ[ x ∈ Fin n ] Rˢ 0 (fsuc x) × Closure R₁ x y
    D' (suc x) zero = Σ[ y ∈ Fin n ] Closure R₁ x y × Rˢ (fsuc y) 0
```

Finally, in order to decide whether $x+1$ and $y+1$ are related by $R^*$,
we have two possibilities: either there is a path from $x$ to $y$ in
$[n]$, which we can find using the induction hypothesis, or there are
are paths from $x+1$ to $0$ and from $0$ to $y+1$ in $[n+1]$, which we
can find using the previous two cases.

```agda
    D' (suc x) (suc y) = Closure R₁ x y ⊎ D' (suc x) zero × D' zero (suc y)
```

<!--
```agda
    D : ∀ i j → Type ℓ
    D i j = D' (fin-view i) (fin-view j)
```
-->

<details>
<summary>
Proving that (the [[propositional truncation]] of) $D$ is a decidable
congruence is tedious but not difficult.

```agda
    D-cong : Congruence (Fin (suc n)) _
    instance D-Dec : ∀ {x y} → Dec (D x y)
```
</summary>

```agda
    D-refl : ∀ x → D x x
    D-refl i with fin-view i
    ... | zero  = _
    ... | suc x = inl R₁.reflᶜ

    D-trans : ∀ x y z → D x y → D y z → D x z
    D-trans i j k p q with fin-view i | fin-view j | fin-view k | p | q
    ... | zero  | zero  | z     | _            | d            = d
    ... | zero  | suc y | zero  | _            | _            = _
    ... | zero  | suc y | suc z | y' , ry , cy | inl c        = y' , ry , cy R₁.∙ᶜ c
    ... | zero  | suc y | suc z | _            | inr (_ , dz) = dz
    ... | suc x | zero  | zero  | d            | _            = d
    ... | suc x | zero  | suc z | dx           | dy           = inr (dx , dy)
    ... | suc x | suc y | zero  | inl c        | y' , cy , ry = y' , c R₁.∙ᶜ cy , ry
    ... | suc x | suc y | zero  | inr (dx , _) | _            = dx
    ... | suc x | suc y | suc z | inl c        | inl d = inl (c R₁.∙ᶜ d)
    ... | suc x | suc y | suc z | inl c        | inr ((y' , cy , ry) , dz) =
      inr ((y' , c R₁.∙ᶜ cy , ry) , dz)
    ... | suc x | suc y | suc z | inr (dx , (y' , ry , cy)) | inl c =
      inr (dx , y' , ry , cy R₁.∙ᶜ c)
    ... | suc x | suc y | suc z | inr (dx , dy) | inr (dy' , dz) =
      inr (dx , dz)

    D-sym : ∀ {i j} (x : Fin-view i) (y : Fin-view j) → D' x y → D' y x
    D-sym zero zero    _            = _
    D-sym zero (suc y) (y' , r , c) = y' , R₁.symᶜ c , ⊎-comm .fst r
    D-sym (suc x) zero (x' , c , r) = x' , ⊎-comm .fst r , R₁.symᶜ c
    D-sym (suc x) (suc y) (inl r)   = inl (R₁.symᶜ r)
    D-sym (suc x) (suc y) (inr (dx , dy)) =
      inr (D-sym zero (suc y) dy , D-sym (suc x) zero dx)

    D-cong ._∼_ x y = ∥ D x y ∥
    D-cong .has-is-prop _ _ = hlevel 1
    D-cong .reflᶜ {x} = inc (D-refl x)
    D-cong ._∙ᶜ_ {x} {y} {z} = ∥-∥-map₂ (D-trans x y z)
    D-cong .symᶜ {x} {y} = map (D-sym (fin-view x) (fin-view y))

    {-# INCOHERENT D-Dec #-}
    D-Dec {i} {j} with fin-view i | fin-view j
    ... | zero  | zero  = auto
    ... | zero  | suc y = auto
    ... | suc x | zero  = auto
    ... | suc x | suc y = auto
```
</details>

To complete the proof, we show that $D$ is indeed equivalent to $R^*$;
it suffices to show that $D$ lies between $R$ and $R^*$.

```agda
    R→D : ∀ {x y} → R x y → D x y
    R→D {i} {j} r with fin-view i | fin-view j
    ... | zero  | zero  = _
    ... | zero  | suc y = y , inl r , R₁.reflᶜ
    ... | suc x | zero  = x , R₁.reflᶜ , inl r
    ... | suc x | suc y = inl (inc r)

    D→R* : ∀ {x y i j} → D' {x} {y} i j → Closure R x y
    D→R* {i = zero}  {j = zero}  _ = R.reflᶜ
    D→R* {i = zero}  {j = suc y} (y' , r , c) = Rˢ→R r R.∙ᶜ R₁→R c
    D→R* {i = suc x} {j = zero}  (x' , c , r) = R₁→R c R.∙ᶜ Rˢ→R r
    D→R* {i = suc x} {j = suc y} (inl r) = R₁→R r
    D→R* {i = suc x} {j = suc y} (inr (dx , dy)) =
      D→R* {i = suc x} {j = zero} dx R.∙ᶜ D→R* {i = zero} {suc y} dy

    R*→D : ∀ {x y} → Closure R x y → ∥ D x y ∥
    R*→D = Closure-rec-congruence R D-cong (inc ∘ R→D)

    R*-dec : Dec (Closure R x y)
    R*-dec = invmap (rec! D→R*) R*→D (holds? ∥ D x y ∥)
```

We can finally assemble the pieces together: given
$f, g : [m] \to [n]$, the coequaliser of $f$ and $g$ is equivalent to
the quotient of $[n]$ by the decidable relation $R$ induced by $f$ and
$g$, and hence by its congruence closure $R^*$. But we know that quotients
of finite sets by decidable congruences are finite, and we just proved
that the closure of a decidable relation on a finite set is decidable,
so we're done.

```agda
Finite-coequaliser
  : ∀ {n m} (f g : Fin m → Fin n)
  → Σ Nat λ q → Fin q ≃ Coeq f g
Finite-coequaliser {n} f g
  = n/R .fst
  , n/R .snd
    ∙e Closure-quotient R e⁻¹
    ∙e Coeq≃quotient f g e⁻¹
  where
    R = span→R f g

    n/R : Σ Nat λ q → Fin q ≃ Fin n / Closure R
    n/R = Finite-quotient (Closure-congruence R)
```
