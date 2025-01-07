<!--
```agda
open import 1Lab.Path
open import 1Lab.Type

open import Data.Nat.Order
open import Data.Dec.Base
open import Data.Nat.Base
open import Data.Sum.Base
open import Data.Bool
```
-->

```agda
module Data.Nat.Properties where
```

# Natural numbers - properties

This module contains proofs of arithmetic identities for [the natural
numbers]. Since they're mostly simple inductive arguments written in
[equational reasoning] style, they are very lightly commented:

[the natural numbers]: Data.Nat.Base.html
[equational reasoning]: 1Lab.Path.html#syntax-sugar

## Addition

```agda
+-associative : (x y z : Nat) → x + (y + z) ≡ (x + y) + z
+-associative zero y z = refl
+-associative (suc x) y z =
  suc (x + (y + z)) ≡⟨ ap suc (+-associative x y z) ⟩
  suc ((x + y) + z) ∎

+-zeror : (x : Nat) → x + 0 ≡ x
+-zeror zero = refl
+-zeror (suc x) =
  suc (x + 0) ≡⟨ ap suc (+-zeror x) ⟩
  suc x       ∎

+-sucr : (x y : Nat) → x + suc y ≡ suc (x + y)
+-sucr zero y = refl
+-sucr (suc x) y = ap suc (+-sucr x y)

+-commutative : (x y : Nat) → x + y ≡ y + x
+-commutative zero y = sym (+-zeror y)
+-commutative (suc x) y =
  suc (x + y) ≡⟨ ap suc (+-commutative x y) ⟩
  suc (y + x) ≡⟨ sym (+-sucr y x) ⟩
  y + suc x   ∎

+-inj : ∀ k x y → k + x ≡ k + y → x ≡ y
+-inj zero x y p = p
+-inj (suc k) x y p = +-inj k x y (suc-inj p)
```

## Multiplication

```agda
*-distrib-+r : (x y z : Nat) → (x + y) * z ≡ x * z + y * z
*-distrib-+r zero y z = refl
*-distrib-+r (suc x) y z =
  z + (x + y) * z     ≡⟨ ap₂ _+_ refl (*-distrib-+r x y z) ⟩
  z + (x * z + y * z) ≡⟨ +-associative z (x * z) (y * z) ⟩
  z + x * z + y * z   ∎

*-sucr : (m n : Nat) → m * suc n ≡ m + m * n
*-sucr zero    n = refl
*-sucr (suc m) n =
  suc m * suc n         ≡⟨⟩
  suc n + m * suc n     ≡⟨ ap₂ _+_ refl (*-sucr m n) ⟩
  suc n + (m + m * n)   ≡⟨⟩
  suc (n + (m + m * n)) ≡⟨ ap suc (+-associative n m (m * n)) ⟩
  suc (n + m + m * n)   ≡⟨ ap (λ x → suc (x + m * n)) (+-commutative n m) ⟩
  suc (m + n + m * n)   ≡˘⟨ ap suc (+-associative m n (m * n)) ⟩
  suc (m + (n + m * n)) ≡⟨⟩
  suc m + suc m * n     ∎

*-onel : (x : Nat) → 1 * x ≡ x
*-onel x = +-zeror x

*-oner : (x : Nat) → x * 1 ≡ x
*-oner zero = refl
*-oner (suc x) =
  suc (x * 1) ≡⟨ ap suc (*-oner x) ⟩
  suc x       ∎

*-zeror : (x : Nat) → x * 0 ≡ 0
*-zeror zero = refl
*-zeror (suc x) = *-zeror x

*-commutative : (x y : Nat) → x * y ≡ y * x
*-commutative zero y    = sym (*-zeror y)
*-commutative (suc x) y =
  y + x * y ≡⟨ ap₂ _+_ refl (*-commutative x y) ⟩
  y + y * x ≡⟨ sym (*-sucr y x) ⟩
  y * suc x ∎

*-distrib-+l : (x y z : Nat) → z * (x + y) ≡ z * x + z * y
*-distrib-+l x y z =
  z * (x + y)   ≡⟨ *-commutative z (x + y) ⟩
  (x + y) * z   ≡⟨ *-distrib-+r x y z ⟩
  x * z + y * z ≡⟨ ap₂ _+_ (*-commutative x z) (*-commutative y z) ⟩
  z * x + z * y ∎

*-associative : (x y z : Nat) → (x * y) * z ≡ x * (y * z)
*-associative zero y z = refl
*-associative (suc x) y z =
  (y + x * y) * z     ≡⟨ *-distrib-+r y (x * y) z ⟩
  y * z + (x * y) * z ≡⟨ ap₂ _+_ refl (*-associative x y z) ⟩
  y * z + x * (y * z) ∎

*-suc-inj : ∀ k x y → x * suc k ≡ y * suc k → x ≡ y
*-suc-inj k zero zero p = refl
*-suc-inj k zero (suc y) p = absurd (zero≠suc p)
*-suc-inj k (suc x) zero p = absurd (suc≠zero p)
*-suc-inj k (suc x) (suc y) p = ap suc (*-suc-inj k x y (+-inj _ _ _ p))

*-suc-inj' : ∀ k x y → suc k * x ≡ suc k * y → x ≡ y
*-suc-inj' k x y p = *-suc-inj k x y (*-commutative x (suc k) ·· p ·· *-commutative (suc k) y)

*-injr : ∀ k x y .⦃ _ : Positive k ⦄ → x * k ≡ y * k → x ≡ y
*-injr (suc k) x y p = *-suc-inj k x y p

*-injl : ∀ k x y .⦃ _ : Positive k ⦄ → k * x ≡ k * y → x ≡ y
*-injl (suc k) x y p = *-suc-inj' k x y p

*-is-onel : ∀ x n → x * n ≡ 1 → x ≡ 1
*-is-onel zero n p = p
*-is-onel (suc zero) zero p = refl
*-is-onel (suc (suc x)) zero p = absurd (zero≠suc (sym (*-zeror x) ∙ p))
*-is-onel (suc x) (suc zero) p = ap suc (sym (*-oner x)) ∙ p
*-is-onel (suc x) (suc (suc n)) p = absurd (zero≠suc (sym (suc-inj p)))

*-is-oner : ∀ x n → x * n ≡ 1 → n ≡ 1
*-is-oner x n p = *-is-onel n x (*-commutative n x ∙ p)

*-is-zero : ∀ x y → x * y ≡ 0 → (x ≡ 0) ⊎ (y ≡ 0)
*-is-zero zero y p = inl refl
*-is-zero (suc x) zero p = inr refl
*-is-zero (suc x) (suc y) p = absurd (suc≠zero p)

*-is-zerol : ∀ x y ⦃ _ : Positive y ⦄ → x * y ≡ 0 → x ≡ 0
*-is-zerol x (suc y) p with *-is-zero x (suc y) p
... | inl p = p
... | inr q = absurd (suc≠zero q)

*-is-zeror : ∀ x y ⦃ _ : Positive x ⦄ → x * y ≡ 0 → y ≡ 0
*-is-zeror (suc x) y p with *-is-zero (suc x) y p
... | inl p = absurd (suc≠zero p)
... | inr q = q
```

## Exponentiation

```agda
^-oner : (x : Nat) → x ^ 1 ≡ x
^-oner x = *-oner x

^-onel : (x : Nat) → 1 ^ x ≡ 1
^-onel zero = refl
^-onel (suc x) =
  (1 ^ x) + 0 ≡⟨ +-zeror (1 ^ x) ⟩
  (1 ^ x)     ≡⟨ ^-onel x ⟩
  1 ∎

^-+-hom-*r : (x y z : Nat) → x ^ (y + z) ≡ (x ^ y) * (x ^ z)
^-+-hom-*r x zero z = sym (+-zeror (x ^ z))
^-+-hom-*r x (suc y) z =
  x * x ^ (y + z)     ≡⟨ ap (x *_) (^-+-hom-*r x y z) ⟩
  x * (x ^ y * x ^ z) ≡⟨ sym (*-associative x (x ^ y) (x ^ z)) ⟩
  x * x ^ y * x ^ z ∎

^-distrib-*r : (x y z : Nat) → (x * y) ^ z ≡ x ^ z * y ^ z
^-distrib-*r x y zero = refl
^-distrib-*r x y (suc z) =
  x * y * (x * y) ^ z     ≡⟨ ap (λ a → x * y * a) (^-distrib-*r x y z) ⟩
  x * y * (x ^ z * y ^ z) ≡⟨ sym (*-associative (x * y) (x ^ z) (y ^ z)) ⟩
  x * y * x ^ z * y ^ z   ≡⟨ ap (_* y ^ z) (*-associative x y (x ^ z)) ⟩
  x * (y * x ^ z) * y ^ z ≡⟨ ap (λ a → x * a * y ^ z) (*-commutative y (x ^ z)) ⟩
  x * (x ^ z * y) * y ^ z ≡⟨ ap (_* y ^ z) (sym (*-associative x (x ^ z) y)) ⟩
  x * x ^ z * y * y ^ z   ≡⟨ *-associative (x * x ^ z) y (y ^ z) ⟩
  x * x ^ z * (y * y ^ z) ∎

^-*-adjunct : (x y z : Nat) → (x ^ y) ^ z ≡ x ^ (y * z)
^-*-adjunct x zero z = ^-onel z
^-*-adjunct x (suc y) zero = ^-*-adjunct x y zero
^-*-adjunct x (suc y) (suc z) =
  x * x ^ y * (x * x ^ y) ^ z       ≡⟨ ap (λ a → x * x ^ y * a) (^-distrib-*r x (x ^ y) z) ⟩
  x * x ^ y * (x ^ z * (x ^ y) ^ z) ≡⟨ ap (λ a → x * x ^ y * (x ^ z * a)) (^-*-adjunct x y z) ⟩
  x * x ^ y * (x ^ z * x ^ (y * z)) ≡⟨ ap (λ a → x * x ^ y * a) (sym (^-+-hom-*r x z (y * z))) ⟩
  x * x ^ y * (x ^ (z + (y * z)))   ≡⟨ *-associative x (x ^ y) (x ^ (z + y * z)) ⟩
  x * (x ^ y * (x ^ (z + (y * z)))) ≡⟨ ap (x *_) (sym (^-+-hom-*r x y (z + y * z))) ⟩
  x * x ^ (y + (z + y * z))         ≡⟨ ap (λ a → x * x ^ a) (+-associative y z (y * z)) ⟩
  x * x ^ (y + z + y * z)           ≡⟨ ap (λ a → x * x ^ (a + y * z)) (+-commutative y z) ⟩
  x * x ^ (z + y + y * z)           ≡˘⟨ ap (λ a → x * x ^ a) (+-associative z y (y * z)) ⟩
  x * x ^ (z + (y + y * z))         ≡⟨ ap (λ a → x * x ^ (z + a)) (sym (*-sucr y z))  ⟩
  x * x ^ (z + y * suc z) ∎
```


### Compatibility

The order relation on the natural numbers is also compatible with the
arithmetic operators:

```agda
+-≤l : (x y : Nat) → x ≤ (x + y)
+-≤l zero y = 0≤x
+-≤l (suc x) y = s≤s (+-≤l x y)

+-≤r : (x y : Nat) → y ≤ (x + y)
+-≤r x zero = 0≤x
+-≤r x (suc y) = subst (λ p → suc y ≤ p) (sym (+-sucr x y)) (s≤s (+-≤r x y))

monus-≤ : (x y : Nat) → x - y ≤ x
monus-≤ x zero = x≤x
monus-≤ zero (suc y) = 0≤x
monus-≤ (suc x) (suc y) = ≤-sucr (monus-≤ x y)

+-preserves-≤l : (x y z : Nat) → x ≤ y → (z + x) ≤ (z + y)
+-preserves-≤l .0 y zero 0≤x = 0≤x
+-preserves-≤l .0 y (suc z) 0≤x =
  s≤s (+-preserves-≤l zero y z 0≤x)
+-preserves-≤l .(suc _) .(suc _) zero (s≤s p) = s≤s p
+-preserves-≤l .(suc _) .(suc _) (suc z) (s≤s p) =
  s≤s (+-preserves-≤l (suc _) (suc _) z (s≤s p))

+-preserves-≤r : (x y z : Nat) → x ≤ y → (x + z) ≤ (y + z)
+-preserves-≤r x y z prf = subst (λ a → a ≤ (y + z)) (+-commutative z x)
  (subst (λ a → (z + x) ≤ a) (+-commutative z y) (+-preserves-≤l x y z prf))

+-preserves-≤ : (x y x' y' : Nat) → x ≤ y → x' ≤ y' → (x + x') ≤ (y + y')
+-preserves-≤ x y x' y' prf prf' = ≤-trans
  (+-preserves-≤r x y x' prf) (+-preserves-≤l x' y' y prf')

+-preserves-<l : (x y z : Nat) → x < y → (z + x) < (z + y)
+-preserves-<l x (suc y) z (s≤s p) = ≤-trans (s≤s (+-preserves-≤l x y z p)) (≤-refl' (sym (+-sucr z y)))

+-preserves-<r : (x y z : Nat) → x < y → (x + z) < (y + z)
+-preserves-<r x y z p = subst₂ _<_ (+-commutative z x) (+-commutative z y) (+-preserves-<l x y z p)

+-preserves-< : ∀ x y x' y' → x < y → x' < y' → (x + x') < (y + y')
+-preserves-< x y x' y' p q = <-trans _ _ _ (+-preserves-<r x y x' p) (+-preserves-<l x' y' y q)

*-preserves-≤l : (x y z : Nat) → x ≤ y → (z * x) ≤ (z * y)
*-preserves-≤l x y zero prf = 0≤x
*-preserves-≤l x y (suc z) prf = +-preserves-≤ x y (z * x) (z * y) prf
  (*-preserves-≤l x y z prf)

*-preserves-≤r : (x y z : Nat) → x ≤ y → (x * z) ≤ (y * z)
*-preserves-≤r x y z prf = subst (λ a → a ≤ (y * z)) (*-commutative z x)
  (subst (λ a → (z * x) ≤ a) (*-commutative z y) (*-preserves-≤l x y z prf))

*-preserves-≤ : (x y x' y' : Nat) → x ≤ y → x' ≤ y' → (x * x') ≤ (y * y')
*-preserves-≤ x y x' y' prf prf' = ≤-trans
  (*-preserves-≤r x y x' prf) (*-preserves-≤l x' y' y prf')

+-reflects-≤l : (x y z : Nat) → (z + x) ≤ (z + y) → x ≤ y
+-reflects-≤l x y zero prf = prf
+-reflects-≤l x y (suc z) (s≤s prf) = +-reflects-≤l x y z prf

+-reflects-≤r : (x y z : Nat) → (x + z) ≤ (y + z) → x ≤ y
+-reflects-≤r x y z prf =
  +-reflects-≤l x y z
    (subst (_≤ (z + y)) (+-commutative x z)
    (subst ((x + z) ≤_) (+-commutative y z) prf))

difference→≤ : ∀ {x z} y → x + y ≡ z → x ≤ z
difference→≤ {x} {z} zero p            = subst (x ≤_) (sym (+-zeror x) ∙ p) ≤-refl
difference→≤ {zero}  {z}     (suc y) p = 0≤x
difference→≤ {suc x} {zero}  (suc y) p = absurd (suc≠zero p)
difference→≤ {suc x} {suc z} (suc y) p = s≤s (difference→≤ (suc y) (suc-inj p))

nonzero→positive : ∀ {x} → x ≠ 0 → 0 < x
nonzero→positive {zero} p = absurd (p refl)
nonzero→positive {suc x} p = s≤s 0≤x

*-cancel-≤r : ∀ x {y z} .⦃ _ : Positive x ⦄ → (y * x) ≤ (z * x) → y ≤ z
*-cancel-≤r (suc x) {zero} {z} p = 0≤x
*-cancel-≤r (suc x) {suc y} {suc z} (s≤s p) = s≤s
  (*-cancel-≤r (suc x) {y} {z} (+-reflects-≤l (y * suc x) (z * suc x) x p))
```

## Monus

```agda
monus-zero : ∀ a → 0 - a ≡ 0
monus-zero zero = refl
monus-zero (suc a) = refl

monus-cancell : ∀ k m n → (k + m) - (k + n) ≡ m - n
monus-cancell zero    = λ _ _ → refl
monus-cancell (suc k) = monus-cancell k

monus-inversel : ∀ n m → m ≤ n → m + (n - m) ≡ n
monus-inversel n m 0≤x = refl
monus-inversel (suc n) (suc m) (s≤s p) = ap suc (monus-inversel n m p)

monus-distribr : ∀ m n k → (m - n) * k ≡ m * k - n * k
monus-distribr m       zero    k = refl
monus-distribr zero    (suc n) k = sym (monus-zero (k + n * k))
monus-distribr (suc m) (suc n) k =
  monus-distribr m n k ∙ sym (monus-cancell k (m * k) (n * k))

monus-distribl : ∀ k m n → k * (m - n) ≡ k * m - k * n
monus-distribl k m n = *-commutative k (m - n) ∙ monus-distribr m n k ∙ ap₂ _-_ (*-commutative m k) (*-commutative n k)

monus-cancelr : ∀ m n k → (m + k) - (n + k) ≡ m - n
monus-cancelr m n k = (λ i → +-commutative m k i - +-commutative n k i) ∙ monus-cancell k m n

monus-addl : ∀ m n k → m - (n + k) ≡ m - n - k
monus-addl zero zero k = refl
monus-addl zero (suc n) k = sym (monus-zero k)
monus-addl (suc m) zero k = refl
monus-addl (suc m) (suc n) k = monus-addl m n k

monus-commute : ∀ m n k → m - n - k ≡ m - k - n
monus-commute m n k =
  m - n - k   ≡˘⟨ monus-addl m n k ⟩
  m - (n + k) ≡⟨ ap (m -_) (+-commutative n k) ⟩
  m - (k + n) ≡⟨ monus-addl m k n ⟩
  m - k - n   ∎

monus-swapl : ∀ x y z → x + y ≡ z → y ≡ z - x
monus-swapl x y z p = sym (monus-cancell x y 0) ∙ ap (x + y -_) (+-zeror x) ∙ ap (_- x) p

monus-inverser : ∀ n m → (m + n) - m ≡ n
monus-inverser n m = sym (monus-swapl m n (m + n) refl)

monus-swapr : ∀ x y z → x + y ≡ z → x ≡ z - y
monus-swapr x y z p = sym (monus-cancelr x 0 y) ∙ ap (_- y) p
```

## Maximum

```agda
max-assoc : (x y z : Nat) → max x (max y z) ≡ max (max x y) z
max-assoc zero zero zero = refl
max-assoc zero zero (suc z) = refl
max-assoc zero (suc y) zero = refl
max-assoc zero (suc y) (suc z) = refl
max-assoc (suc x) zero zero = refl
max-assoc (suc x) zero (suc z) = refl
max-assoc (suc x) (suc y) zero = refl
max-assoc (suc x) (suc y) (suc z) = ap suc (max-assoc x y z)

max-≤l : (x y : Nat) → x ≤ max x y
max-≤l zero zero = 0≤x
max-≤l zero (suc y) = 0≤x
max-≤l (suc x) zero = ≤-refl
max-≤l (suc x) (suc y) = s≤s (max-≤l x y)

max-≤r : (x y : Nat) → y ≤ max x y
max-≤r zero zero = 0≤x
max-≤r zero (suc y) = ≤-refl
max-≤r (suc x) zero = 0≤x
max-≤r (suc x) (suc y) = s≤s (max-≤r x y)

max-univ : (x y z : Nat) → x ≤ z → y ≤ z → max x y ≤ z
max-univ zero zero z 0≤x 0≤x = 0≤x
max-univ zero (suc y) (suc z) 0≤x (s≤s q) = s≤s q
max-univ (suc x) zero (suc z) (s≤s p) 0≤x = s≤s p
max-univ (suc x) (suc y) (suc z) (s≤s p) (s≤s q) = s≤s (max-univ x y z p q)

max-zerol : (x : Nat) → max 0 x ≡ x
max-zerol zero = refl
max-zerol (suc x) = refl

max-zeror : (x : Nat) → max x 0 ≡ x
max-zeror zero = refl
max-zeror (suc x) = refl
```

## Minimum

```agda
min-assoc : (x y z : Nat) → min x (min y z) ≡ min (min x y) z
min-assoc zero zero zero = refl
min-assoc zero zero (suc z) = refl
min-assoc zero (suc y) zero = refl
min-assoc zero (suc y) (suc z) = refl
min-assoc (suc x) zero zero = refl
min-assoc (suc x) zero (suc z) = refl
min-assoc (suc x) (suc y) zero = refl
min-assoc (suc x) (suc y) (suc z) = ap suc (min-assoc x y z)

min-≤l : (x y : Nat) → min x y ≤ x
min-≤l zero zero = 0≤x
min-≤l zero (suc y) = 0≤x
min-≤l (suc x) zero = 0≤x
min-≤l (suc x) (suc y) = s≤s (min-≤l x y)

min-≤r : (x y : Nat) → min x y ≤ y
min-≤r zero zero = 0≤x
min-≤r zero (suc y) = 0≤x
min-≤r (suc x) zero = 0≤x
min-≤r (suc x) (suc y) = s≤s (min-≤r x y)

min-univ : (x y z : Nat) → z ≤ x → z ≤ y → z ≤ min x y
min-univ x y zero 0≤x 0≤x = 0≤x
min-univ (suc x) (suc y) (suc z) (s≤s p) (s≤s q) = s≤s (min-univ x y z p q)

min-zerol : (x : Nat) → min 0 x ≡ 0
min-zerol zero = refl
min-zerol (suc x) = refl

min-zeror : (x : Nat) → min x 0 ≡ 0
min-zeror zero = refl
min-zeror (suc x) = refl
```

## The factorial function

```agda
factorial-positive : ∀ n → Positive (factorial n)
factorial-positive zero = s≤s 0≤x
factorial-positive (suc n) = *-preserves-≤ 1 (suc n) 1 (factorial n) (s≤s 0≤x) (factorial-positive n)

≤-factorial : ∀ n → n ≤ factorial n
≤-factorial zero = 0≤x
≤-factorial (suc n) = subst (_≤ factorial (suc n)) (*-oner (suc n)) (*-preserves-≤ (suc n) (suc n) 1 (factorial n) ≤-refl (factorial-positive n))
```
