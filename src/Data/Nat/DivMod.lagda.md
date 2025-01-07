<!--
```agda
open import 1Lab.Prelude

open import Data.Nat.Properties
open import Data.Nat.Divisible
open import Data.Nat.Solver
open import Data.Nat.Order
open import Data.Dec.Base
open import Data.Nat.Base
open import Data.Sum.Base
open import Data.Irr
```
-->

```agda
module Data.Nat.DivMod where
```

# Natural division {defines="natural-division"}

This module implements the basics of the theory of **division** (not
[divisibility], see there) for the natural numbers. In particular, we
define what it means to divide in the naturals (the type
`DivMod`{.Agda}), and implement a division procedure that works for
positive denominators.

[divisibility]: Data.Nat.Divisible.html

The result of division isn't a single number, but rather a **pair** of
numbers $q, r$ such that $a = qb + r$. The number $q$ is the
**quotient** $a /_n b$, and the number $r$ is the **remainder**
$a \% b$.

```agda
record DivMod (a b : Nat) : Type where
  constructor divmod

  field
    quot : Nat
    rem  : Nat
    .quotient : a ≡ quot * b + rem
    .smaller  : rem < b
```

The easy case is to divide zero by anything, in which case the result is
zero with remainder zero. The more interesting case comes when we have
some successor, and we want to divide it.

<!--
```agda
module _ where private
-- This is a nice explanation of the division algorithm but it ends up
-- being extremely slow when compared to the builtins. So we define the
-- nice prosaic version first, then define the fast version hidden..
```
-->

```agda
  divide-pos : ∀ a b → .⦃ _ : Positive b ⦄ → DivMod a b
  divide-pos zero (suc b) = divmod 0 0 refl (s≤s 0≤x)
```

It suffices to assume --- since $a$ is smaller than $1+a$ --- that we
have already computed numbers $q', r'$ with $a = q'b + r'$. Since the
ordering on $\NN$ is trichotomous, we can proceed by cases on whether
$(1 + r') < b$.

```agda
  divide-pos (suc a) b with divide-pos a b
  divide-pos (suc a) b | divmod q' r' p s with ≤-split (suc r') b
```

First, suppose that $1 + r' < b$, i.e., $1 + r'$ _can_ serve as a
remainder for the division of $(1 + a) / b$. In that case, we have our
divisor! It remains to show, by a slightly annoying computation, that

$$
(1 + a) = q'b + 1 + r
$$

```agda
  divide-pos (suc a) b | divmod q' r' p s | inl r'+1<b =
    divmod q' (suc r') (ap suc p ∙ sym (+-sucr (q' * b) r')) r'+1<b
```

The other case --- that in which $1 + r' = b$ --- is more interesting.
Then, rather than incrementing the remainder, our remainder has
"overflown", and we have to increment the _quotient_ instead. We take,
in this case, $q = 1 + q'$ and $r = 0$, which works out because ($0 < b$
and) of some arithmetic. See for yourself:

```agda
  divide-pos (suc a) (suc b') | divmod q' r' p s | inr (inr r'+1=b) =
    divmod (suc q') 0
      ( suc a                           ≡⟨ ap suc p ⟩
        suc (q' * (suc b') + r')        ≡˘⟨ ap (λ e → suc (q' * e + r')) r'+1=b ⟩
        suc (q' * (suc r') + r')        ≡⟨ nat! ⟩
        suc (r' + q' * (suc r') + zero) ≡⟨ ap (λ e → e + q' * e + 0) r'+1=b ⟩
        (suc b') + q' * (suc b') + 0    ∎ )
      (s≤s 0≤x)
```

Note that we actually have one more case to consider -- or rather,
discard -- that in which $b < 1 + r'$. It's impossible because, by the
definition of division, we have $r' < b$, meaning $(1 + r') \le b$.

```agda
  divide-pos (suc a) (suc b') | divmod q' r' p s | inr (inl b<r'+1) =
    absurd (<-not-equal b<r'+1
      (≤-antisym (≤-sucr (≤-peel b<r'+1)) (recover s)))
```

As a finishing touch, we define short operators to produce the result of
applying `divide-pos`{.Agda} to a pair of numbers. Note that the
procedure above only works when the denominator is nonzero, so we have a
degree of freedom when defining $x/0$ and $x \% 0$. The canonical choice
is to yield $0$ in both cases.

```agda
  _%_ : Nat → Nat → Nat
  a % zero = zero
  a % suc b = divide-pos a (suc b) .DivMod.rem

  _/ₙ_ : Nat → Nat → Nat
  a /ₙ zero = zero
  a /ₙ suc b = divide-pos a (suc b) .DivMod.quot

  is-divmod : ∀ x y → .⦃ _ : Positive y ⦄ → x ≡ (x /ₙ y) * y + x % y
  is-divmod x (suc y) with divide-pos x (suc y)
  ... | divmod q r α β = recover α

  x%y<y : ∀ x y → .⦃ _ : Positive y ⦄ → (x % y) < y
  x%y<y x (suc y) with divide-pos x (suc y)
  ... | divmod q r α β = recover β
```

With this, we can decide whether two numbers divide each other by
checking whether the remainder is zero!

<!--
```agda
mod-helper : (k m n j : Nat) → Nat
mod-helper k m  zero    j      = k
mod-helper k m (suc n)  zero   = mod-helper 0       m n m
mod-helper k m (suc n) (suc j) = mod-helper (suc k) m n j

div-helper : (k m n j : Nat) → Nat
div-helper k m  zero    j      = k
div-helper k m (suc n)  zero   = div-helper (suc k) m n m
div-helper k m (suc n) (suc j) = div-helper k       m n j

{-# BUILTIN NATDIVSUCAUX div-helper #-}
{-# BUILTIN NATMODSUCAUX mod-helper #-}

-- _ = {! mod-helper 0 0 4294967296 2  !}

_/ₙ_ : (d1 d2 : Nat) .⦃ _ : Positive d2 ⦄ → Nat
d1 /ₙ suc d2 = div-helper 0 d2 d1 d2

_%_ : (d1 d2 : Nat) .⦃ _ : Positive d2 ⦄ → Nat
d1 % suc d2 = mod-helper 0 d2 d1 d2

infixl 9 _/ₙ_ _%_

abstract
  private
    div-mod-lemma : ∀ am ad d n → am + ad * suc (am + n) + d ≡ mod-helper am (am + n) d n + div-helper ad (am + n) d n * suc (am + n)
    div-mod-lemma am ad zero n = +-zeror _
    div-mod-lemma am ad (suc d) zero rewrite Id≃path.from (+-zeror am) = +-sucr _ d ∙ div-mod-lemma zero (suc ad) d am
    div-mod-lemma am ad (suc d) (suc n) rewrite Id≃path.from (+-sucr am n) = +-sucr _ d ∙ div-mod-lemma (suc am) ad d n

    mod-le : ∀ a d n → mod-helper a (a + n) d n ≤ a + n
    mod-le a zero n = +-≤l a n
    mod-le a (suc d) zero = mod-le zero d (a + 0)
    mod-le a (suc d) (suc n) rewrite Id≃path.from (+-sucr a n) = mod-le (suc a) d n

  is-divmod : ∀ x y → .⦃ _ : Positive y ⦄ → x ≡ (x /ₙ y) * y + x % y
  is-divmod x (suc y) = div-mod-lemma 0 0 x y ∙ +-commutative (x % (suc y)) _

  x%y<y : ∀ x y → .⦃ _ : Positive y ⦄ → (x % y) < y
  x%y<y x (suc y) = s≤s (mod-le 0 x y)

from-fast-mod : ∀ d1 d2 .⦃ _ : Positive d2 ⦄ → d1 % d2 ≡ 0 → d2 ∣ d1
from-fast-mod d1 d2 p = fibre→∣ (d1 /ₙ d2 , sym (is-divmod d1 d2 ∙ ap (d1 /ₙ d2 * d2 +_) p ∙ +-zeror _))

divide-pos : ∀ a b → .⦃ _ : Positive b ⦄ → DivMod a b
divide-pos a b = record
  { quot = a /ₙ b
  ; rem  = a % b
  ; quotient = is-divmod a b
  ; smaller = x%y<y a b
  }

private module _ where
  open DivMod
  divmod-ap : ∀ {a b} {x y : DivMod a b} → x .quot ≡ y .quot → x .rem ≡ y .rem → x ≡ y
  divmod-ap {a} {b} {divmod _ _ α β} {divmod _ _ α' β'} p q =
    ap {A = Σ[ q ∈ Nat ] Σ[ r ∈ Nat ] (Irr (a ≡ q * b + r) × Irr (r < b))} (λ (w , x , forget y , forget z) → divmod w x y z)
      {x = _ , _ , forget α , forget β} {y = _ , _ , forget α' , forget β'}
      (p ,ₚ q ,ₚ prop! ,ₚ prop!)

instance
  H-Level-DivMod : ∀ {a b n} .⦃ _ : Positive b ⦄ → H-Level (DivMod a b) (suc n)
  H-Level-DivMod {a} {b@(suc b')} =
    prop-instance λ where
      (divmod q r α β) (divmod q' r' α' β') → case ≤-split r r' of λ where
        (inr (inr r=r')) → divmod-ap (*-injr b q q' (+-inj r (q * b) (q' * b) (+-commutative r (q * b) ∙ sym (recover α) ∙ recover α' ∙ +-commutative (q' * b) r' ∙ ap (_+ q' * b) (sym r=r')))) r=r'
        (inr (inl r'<r)) → absurd (lemma' q q' b r r' r'<r (recover β) (recover (sym α ∙ α')))
        (inl r<r')       → absurd (lemma' q' q b r' r r<r' (recover β') (recover (sym α' ∙ α)))
    where
      lemma : ∀ x y z → x < y → x ≡ y * z → z ≡ 0
      lemma x y zero p q = refl
      lemma zero (suc y) (suc z) (s≤s p) q = absurd (zero≠suc q)
      lemma (suc x) (suc y) (suc z) (s≤s p) q = absurd (¬sucx≤x y (≤-trans r' (≤-trans (≤-refl' (sym q)) p))) where
        r : z + y * suc z ≡ y + (z + y * z)
        r = nat!

        r' : suc y ≤ suc (z + y * suc z)
        r' = s≤s (≤-trans (+-≤l y (z + y * z)) (≤-refl' (sym r)))

      lemma' : ∀ q q' b r r' → r' < r → r < b → q * b + r ≡ q' * b + r' → ⊥
      lemma' q q' b r r' r'<r r<b β =
        let
          γ : r' + (r - r') ≡ r
          γ = monus-inversel _ _ (<-weaken r'<r)

          p : (q * b + (r - r')) + r' ≡ q' * b + r'
          p = sym (+-associative (q * b) (r - r') r') ∙ ap (q * b +_) (+-commutative (r - r') r' ∙ γ) ∙ β

          p' : q * b + (r - r') ≡ q' * b
          p' = +-inj r' _ _ (+-commutative r' _ ∙ p ∙ +-commutative _ r')

          p'' : b * (q' - q) + r' ≡ r
          p'' =
            ap (_+ r') (monus-distribl b q' q ∙ ap₂ _-_ (*-commutative b q' ∙ sym p') (*-commutative b q) ∙ monus-inverser (r - r') (q * b))
            ∙ +-commutative (r - r') r' ∙ γ

          d0 : r - r' < b
          d0 = ≤-trans (s≤s (monus-≤ r r')) r<b

          d1 : r - r' ≡ b * (q' - q)
          d1 = sym (monus-swapr (b * (q' - q)) r' r p'')

          d2 : q' - q ≡ 0
          d2 = lemma _ _ (q' - q) d0 d1
        in <-not-equal r'<r (sym (ap (_+ r') (*-zeror b)) ∙ ap (λ e → b * e + r') (sym d2) ∙ p'')

```
-->

```agda
instance
  Dec-∣ : ∀ {n m} → Dec (n ∣ m)
  Dec-∣ {zero} {m} = m ≡? 0
  Dec-∣ n@{suc _} {m} with divide-pos m n
  ... | divmod q 0 α β = yes (q , sym (+-zeror _) ∙ sym (recover α))
  ... | divmod q r@(suc _) α β = no λ (q' , p) →
    let
      n∣r : (q' - q) * n ≡ r
      n∣r = monus-distribr q' q n ∙ sym (monus-swapl _ _ _ (sym (p ∙ recover α)))
    in <-≤-asym (recover β) (m∣sn→m≤sn (q' - q , recover n∣r))
```
