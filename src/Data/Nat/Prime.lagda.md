<!--
```agda
open import 1Lab.Prelude

open import Data.Wellfounded.Properties
open import Data.Nat.Divisible.GCD
open import Data.List.Quantifiers
open import Data.Wellfounded.Base
open import Data.List.Membership
open import Data.Fin.Properties
open import Data.Nat.Properties
open import Data.Nat.Divisible
open import Data.Nat.DivMod
open import Data.List.Base
open import Data.Nat.Order
open import Data.Fin.Base renaming (_≤_ to _≤ᶠ_) hiding (_<_)
open import Data.Nat.Base
open import Data.Dec
open import Data.Irr
open import Data.Sum
```
-->

```agda
module Data.Nat.Prime where
```

# Prime and composite numbers

A number $n$ is **prime** when it is greater than two (which we split
into being positive and being apart from 1), and, if $d$ is any other
number that divides $n$, then either $d = 1$ or $d = n$. Since we've
assumed $n \neq 1$, these cases are disjoint, so being a prime is a
[[proposition]].

```agda
record is-prime (n : Nat) : Type where
  field
    ⦃ positive ⦄ : Positive n
    prime≠1   : n ≠ 1
    primality : ∀ d → d ∣ n → (d ≡ 1) ⊎ (d ≡ n)
```

We note that a prime number has no *proper divisors*, in that, if $d
\neq 1$ and $d \neq n$, then $d$ does not divide $n$.

```agda
  ¬proper-divisor : ∀ {d} → d ≠ 1 → d ≠ n → ¬ d ∣ n
  ¬proper-divisor ¬d=1 ¬d=n d∣n with primality _ d∣n
  ... | inl d=1 = ¬d=1 d=1
  ... | inr d=n = ¬d=n d=n
```

<!--
```agda
  prime>1 : 1 < n
  prime>1 = <-from-≤ (prime≠1 ∘ sym) positive

private unquoteDecl eqv = declare-record-iso eqv (quote is-prime)
open is-prime

abstract instance
  H-Level-is-prime : ∀ {n k} → H-Level (is-prime n) (suc k)
  H-Level-is-prime = prop-instance $ Iso→is-hlevel 1 eqv $ Σ-is-hlevel 1 (hlevel 1) λ p →
    Σ-is-hlevel 1 (hlevel 1) λ q → Π-is-hlevel² 1 λ d dn →
      disjoint-⊎-is-prop (hlevel 1) (hlevel 1) (λ (a , b) → q (sym b ∙ a))
```
-->

Conversely, if $n > 2$ is a number that is not prime, it is called
**composite**. Instead of defining `is-composite`{.Agda} to be the
literal negation of `is-prime`{.Agda}, we instead define this notion
positively: To say that a number is composite is to give a prime $p$ and
a number $q \ge 2$ such that $qp = n$ and $p$ is the smallest divisor of
$n$.

```agda
record is-composite (n : Nat) : Type where
  field
    {p q} : Nat

    p-prime  : is-prime p
    q-proper : 1 < q

    factors : q * p ≡ n
    least : ∀ p' → 1 < p' → p' ∣ n → p ≤ p'
```

Note that the assumption that $p$ is a prime is simply for convenience:
the least proper divisor of *any* number is always a prime.

```agda
least-divisor→is-prime
  : ∀ p n → 1 < p → p ∣ n → (∀ p' → 1 < p' → p' ∣ n → p ≤ p') → is-prime p
least-divisor→is-prime p n gt div least .positive = ≤-trans ≤-ascend gt
least-divisor→is-prime p n gt div least .prime≠1 q = <-irrefl (sym q) gt
least-divisor→is-prime p@(suc p') n gt div least .primality 0 x = absurd (suc≠zero x)
least-divisor→is-prime p@(suc p') n gt div least .primality 1 x = inl refl
least-divisor→is-prime p@(suc p') n gt div least .primality m@(suc (suc k)) x =
  inr (≤-antisym (m∣sn→m≤sn x) (least m (s≤s (s≤s 0≤x)) (∣-trans x div)))
```

<!--
```agda
open is-composite hiding (p ; q ; factors)

private unquoteDecl eqv' = declare-record-iso eqv' (quote is-composite)

abstract instance
  H-Level-is-composite : ∀ {n k} → H-Level (is-composite n) (suc k)
  H-Level-is-composite = prop-instance λ a@record{p = p ; q = q} b@record{p = p' ; q = q'} →
    let
      open is-composite using (factors)

      ap=bp : p ≡ p'
      ap=bp = ≤-antisym
        (a .least p' (prime>1 (b .p-prime)) (fibre→∣ (q' , b .factors)))
        (b .least p (prime>1 (a .p-prime)) (fibre→∣ (q , a .factors)))

      aq=bq : q ≡ q'
      aq=bq = *-injl p q q' (*-commutative p q ∙ a .factors ∙ sym (b .factors) ∙ ap (q' *_) (sym ap=bp) ∙ *-commutative q' p)
    in Equiv.injective (Iso→Equiv eqv') (Σ-pathp ap=bp (Σ-prop-pathp! aq=bq))

prime-divisor-lt : ∀ p q n .⦃ _ : Positive n ⦄ → is-prime p → q * p ≡ n → q < n
prime-divisor-lt p q n pprime div with ≤-strengthen (m∣n→m≤n {q} {n} (fibre→∣ (p , *-commutative p q ∙ div)))
... | inr less = less
... | inl same = absurd (pprime .prime≠1 $
  *-injr n p 1 (sym (+-zeror n ∙ sym div ∙ *-commutative q p ∙ ap (p *_) same)))

prime-remainder-positive : ∀ p q n .⦃ _ : Positive n ⦄ → ¬ is-prime n → is-prime p → q * p ≡ n → 1 < q
prime-remainder-positive p 0 n@(suc _) _ _ div = absurd (zero≠suc div)
prime-remainder-positive p 1 n@(suc _) nn pp div = absurd (nn (subst is-prime (sym (+-zeror p) ∙ div) pp))
prime-remainder-positive p (suc (suc _)) (suc _) _ _ _ = s≤s (s≤s 0≤x)

distinct-primes→coprime : ∀ {a b} → is-prime a → is-prime b → ¬ a ≡ b → is-gcd a b 1
distinct-primes→coprime {a@(suc a')} {b@(suc b')} apr bpr a≠b = record
  { gcd-∣l = a , *-oner a
  ; gcd-∣r = b , *-oner b
  ; greatest = λ {g'} w2 w → case bpr .primality _ w of λ where
    (inl p) → subst (_∣ 1) (sym p) ∣-refl
    (inr q) → case apr .primality _ w2 of λ where
      (inl p) → subst (_∣ 1) (sym p) ∣-refl
      (inr r) → absurd (a≠b (sym r ∙ q))
  }
```
-->

## Primality testing

```agda
is-prime-or-composite : ∀ n → 1 < n → is-prime n ⊎ is-composite n
is-prime-or-composite n@(suc (suc m)) (s≤s p)
  with Fin-omniscience {n = n} (λ k → 1 < k .lower × k .lower ∣ n)
... | inr prime = inl record { prime≠1 = suc≠zero ∘ suc-inj ; primality = no-divisors→prime } where
  no-divisors→prime : ∀ d → d ∣ n → d ≡ 1 ⊎ d ≡ n
  no-divisors→prime d div with d ≡? 1
  ... | yes p = inl p
  ... | no ¬d=1 with d ≡? n
  ... | yes p = inr p
  ... | no ¬d=n =
    let
      ord : d < n
      ord = proper-divisor-< ¬d=n div

      no = case d return (λ x → ¬ x ≡ 1 → x ∣ n → 1 < x) of λ where
        0 p1 div → absurd (<-irrefl (sym div) (s≤s 0≤x))
        1 p1 div → absurd (p1 refl)
        (suc (suc n)) p1 div → s≤s (s≤s 0≤x)
    in absurd (prime (from-ℕ< (d , ord)) (no ¬d=1 div , div))
... | inl (ix , (proper , div) , least)
  = inr record { p-prime = prime ; q-proper = proper' ; factors = path ; least = least' } where
  open Σ (∣→fibre div) renaming (fst to quot ; snd to path)

  abstract
    least' : (p' : Nat) → 1 < p' → p' ∣ n → ix .lower ≤ p'
    least' p' x div with ≤-strengthen (m∣n→m≤n div)
    ... | inl same = ≤-trans ≤-ascend (subst (ix .lower <_) (sym same) (to-ℕ< ix .snd))
    ... | inr less = least (from-ℕ< (p' , less)) (x , div)

  prime : is-prime (ix .lower)
  prime = least-divisor→is-prime (ix .lower) n proper div least'

  proper' : 1 < quot
  proper' with quot | path
  ... | 0 | q = absurd (<-irrefl q (s≤s 0≤x))
  ... | 1 | q = absurd (<-irrefl (sym (+-zeror (ix .lower)) ∙ q) (to-ℕ< ix .snd))
  ... | suc (suc n) | p = s≤s (s≤s 0≤x)

record Factorisation (n : Nat) : Type where
  constructor factored
  field
    primes    : List Nat
    factors   : product primes ≡ n
    is-primes : All is-prime primes

  prime-divides : ∀ {x} → x ∈ₗ primes → x ∣ n
  prime-divides {x} h = subst (x ∣_) factors (work _ _ h) where
    work : ∀ x xs → x ∈ₗ xs → x ∣ product xs
    work x (y ∷ xs) (here p) = substᵢ (λ e → x ∣ e * product xs) p (∣-*l {x} {product xs})
    work x (y ∷ xs) (there p) =
      let
        it = work x xs p
        (div , p) = ∣→fibre it
      in fibre→∣ (y * div , *-associative y div x ∙ ap (y *_) p)

  find-prime-factor : ∀ {x} → is-prime x → x ∣ n → x ∈ₗ primes
  find-prime-factor {num} x d =
    work _ primes x (λ _ → all-∈ is-primes) (subst (num ∣_) (sym factors) d)
    where
    work : ∀ x xs → is-prime x → (∀ x → x ∈ₗ xs → is-prime x) → x ∣ product xs → x ∈ₗ xs
    work x [] xp ps xd = absurd (prime≠1 xp (∣-1 xd))
    work x (y ∷ xs) xp ps xd with x ≡ᵢ? y
    ... | yes x=y = here x=y
    ... | no ¬p = there $ work x xs xp (λ x w → ps x (there w))
      (|-*-coprime-cancel x y (product xs)
        ⦃ auto ⦄ ⦃ ps y (here reflᵢ) .positive ⦄
        xd (distinct-primes→coprime xp (ps y (here reflᵢ)) (¬p ∘ Id≃path.from)))

open is-composite using (factors)
open Factorisation

factorisation-unique' : ∀ {n} (a b : Factorisation n) → ∀ p → p ∈ₗ a .primes → p ∈ₗ b .primes
factorisation-unique' a b p p∈a =
  let
    p∣n = prime-divides a p∈a
  in find-prime-factor b (all-∈ (a .is-primes) p∈a) p∣n

factorisation-worker
  : ∀ n → (∀ k → k < n → .⦃ _ : Positive k ⦄ → Factorisation k)
  → .⦃ _ : Positive n ⦄ → Factorisation n
factorisation-worker 1 ind = factored [] refl []
factorisation-worker n@(suc (suc m)) ind with is-prime-or-composite n (s≤s (s≤s 0≤x))
... | inl prime     = factored (n ∷ []) (ap (2 +_) (*-oner m)) (prime ∷ [])
... | inr composite@record{p = p; q = q} =
  let
    instance
      _ : Positive q
      _ = ≤-trans ≤-ascend (composite .q-proper)

    factored ps path primes = ind q $
      prime-divisor-lt p q n (composite .p-prime) (composite .factors)
  in record
    { primes    = p ∷ ps
    ; factors   = ap (p *_) path ·· *-commutative p q ·· composite .factors
    ; is-primes = composite .p-prime ∷ primes
    }

∣-factorial : ∀ n {m} → m < n → suc m ∣ factorial n
∣-factorial (suc n) {m} m≤n with suc m ≡? suc n
... | yes m=n = subst (_∣ factorial (suc n)) (sym m=n) (∣-*l {suc n} {factorial n})
... | no m≠n  = |-*l-pres {suc m} {suc n} {factorial n} $
  ∣-factorial n {m} (≤-uncap (suc m) n m≠n m≤n)

factorise : (n : Nat) .⦃ _ : Positive n ⦄ → Factorisation n
factorise = Wf-induction _<_ <-wf _ factorisation-worker

new-prime : (n : Nat) → Σ[ p ∈ Nat ] n < p × is-prime p
new-prime n with is-prime-or-composite (suc (factorial n)) (s≤s (factorial-positive n))
new-prime n | inl prime     = suc (factorial n) , s≤s (≤-factorial n) , prime
new-prime n | inr c@record{p = p} with holds? (suc n ≤ p)
new-prime n | inr c@record{p = p} | yes n<p = p , n<p , c .p-prime
new-prime n | inr c@record{p = suc p; q = q} | no ¬n<p =
  let
    rem = ∣-factorial n (<-from-not-≤ _ _ (¬n<p ∘ s≤s))
    rem' : suc p ≡ 1
    rem' = ∣-1 (∣-+-cancel' {n = suc p} {a = 1} {b = factorial n} rem (fibre→∣ (q , c .factors)))
  in absurd (c .p-prime .prime≠1 rem')
```
