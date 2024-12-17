<!--
```agda
open import 1Lab.HLevel.Closure
open import 1Lab.HLevel
open import 1Lab.Path
open import 1Lab.Type

open import Data.Fin.Base
```
-->

```agda
module Data.Fin.Product where
```

# Finitary dependent products

This module defines the product of a finite sequence of types, along
with strictly curried (non-dependent) functions whose domain is a finite
product. The construction is maximally universe-polymorphic, in that it
supports sequences whose universe level varies between components, and
is valued in their finite supremum, giving universes for products more
precise than `Setω`{.Agda}.

<!--
```agda
ℓ-maxᶠ : ∀ {n} (ℓ : Fin n → Level) → Level
ℓ-maxᶠ {n = zero} ℓ = lzero
ℓ-maxᶠ {n = suc n} ℓ = ℓ fzero ⊔ ℓ-maxᶠ (λ i → ℓ (fsuc i))
```
-->

We define the product of a sequence $P$, $\Pi^f P$, by recursion on the
number of elements: the empty product is the unit type by decree, and
the product of a nonempty sequence is the (binary) product of its head
and the $n$-ary product of its tail.

```agda
Πᶠ : ∀ {n ℓ} (P : (i : Fin n) → Type (ℓ i)) → Type (ℓ-maxᶠ ℓ)
Πᶠ {n = 0} P     = ⊤
Πᶠ {n = suc n} P = P fzero × Πᶠ λ i → P (fsuc i)
```

Note that we can define functions converting between the types $\Pi^f P$
and $\prod_{i : [n]} P(i)$. However, since this latter type lives in a
limit universe when $P$ has non-constant level, we can not express that
these functions are isomorphisms in full generality.

```agda
indexₚ : ∀ {n ℓ} {P : (i : Fin n) → Type (ℓ i)}
       → Πᶠ {ℓ = ℓ} P → ∀ i → P i
indexₚ {n = zero}  {P = P} prod ()
indexₚ {n = suc n} {P = P} prod fzero = prod .fst
indexₚ {n = suc n} {P = P} prod (fsuc x) = indexₚ (prod .snd) x

tabulateₚ
  : ∀ {n} {ℓ : Fin n → Level} {P : (i : Fin n) → Type (ℓ i)}
  → (∀ i → P i) → Πᶠ P
tabulateₚ {n = zero} f  = tt
tabulateₚ {n = suc n} f = f fzero , tabulateₚ λ i → f (fsuc i)
```

<!--
```agda
extₚ
  : ∀ {n} {ℓ : Fin n → Level} {P : (i : Fin n) → Type (ℓ i)} {xs ys : Πᶠ P}
  → (∀ i → indexₚ xs i ≡ indexₚ ys i)
  → xs ≡ ys
extₚ {zero} p = refl
extₚ {suc n} p = ap₂ _,_ (p fzero) (extₚ {n} (λ i → p (fsuc i)))
```
-->

Elements of $\Pi^f$ for sequences with a known length enjoy strong
extensionality properties, since they are iterated types with
$\eta$-expansion. As an example:

<!--
```agda
module _ {ℓ : Fin 3 → Level} {P : (i : Fin 3) → Type (ℓ i)} where
```
-->

```agda
  _ : {p : Πᶠ {n = 3} P} → p ≡ (p .fst , p .snd .fst , p .snd .snd .fst , tt)
  _ = refl
```

Exploiting this structure, we can establish that a product has h-level
$k$ when all of its factors do.

```agda
Πᶠ-is-hlevel
  : ∀ {n ℓ} {P : (i : Fin n) → Type (ℓ i)} k
  → (∀ i → is-hlevel (P i) k) → is-hlevel (Πᶠ {ℓ = ℓ} P) k
Πᶠ-is-hlevel {n = 0} {P = P} zero hl    = hlevel 0
Πᶠ-is-hlevel {n = 0} {P = P} (suc k) hl = is-prop→is-hlevel-suc (λ _ _ _ → tt)

Πᶠ-is-hlevel {n = suc n} {P = P} k hl = ×-is-hlevel k (hl fzero) $
  Πᶠ-is-hlevel {P = λ i → P (fsuc i)} k (λ i → hl (fsuc i))
```

More concretely, we can treat these products as data structures, and
update a single value, by index; Or alter the entire tuple using a
sequence of functions. Moreover, these functions have their expected
behaviour, and, when the length of the sequence is a numeral, will
simply compute away.

```agda
updateₚ
  : ∀ {n} {ℓ : Fin n → Level} {P : (i : Fin n) → Type (ℓ i)}
  → Πᶠ P → ∀ i → P i → Πᶠ P
updateₚ xs fzero x    = x , xs .snd
updateₚ xs (fsuc k) x = xs .fst , updateₚ (xs .snd) k x

mapₚ
  : ∀ {n} {ℓ ℓ' : Fin n → Level}
      {P : (i : Fin n) → Type (ℓ i)}
      {Q : (i : Fin n) → Type (ℓ' i)}
  → (∀ i → P i → Q i) → Πᶠ P → Πᶠ Q
mapₚ {0}     f xs = xs
mapₚ {suc n} f xs = f fzero (xs .fst) , mapₚ (λ i → f (fsuc i)) (xs .snd)
```

<!--
```agda
indexₚ-mapₚ
  : ∀ {n} {ℓ ℓ' : Fin n → Level}
      {P : (i : Fin n) → Type (ℓ i)}
      {Q : (i : Fin n) → Type (ℓ' i)}
  → ∀ (f : ∀ i → P i → Q i) (xs : Πᶠ P) i
  → indexₚ (mapₚ f xs) i ≡ f i (indexₚ xs i)
indexₚ-mapₚ {suc n} f xs fzero = refl
indexₚ-mapₚ {suc n} f xs (fsuc i) = indexₚ-mapₚ (λ i → f (fsuc i)) (xs .snd) i

indexₚ-tabulateₚ
  : ∀ {n} {ℓ : Fin n → Level} {P : (i : Fin n) → Type (ℓ i)} (f : ∀ i → P i) i
  → indexₚ (tabulateₚ f) i ≡ f i
indexₚ-tabulateₚ f fzero = refl
indexₚ-tabulateₚ f (fsuc i) = indexₚ-tabulateₚ (λ i → f (fsuc i)) i
```
-->

More generically, we can characterise the entries of an updated product
type.

```agda
updatedₚ
 : ∀ {n} {ℓ : Fin (suc n) → Level} {P : (i : Fin (suc n)) → Type (ℓ i)}
 → (p : Πᶠ P) (i : Fin (suc n)) (x : P i)
 → (indexₚ {P = P} (updateₚ {P = P} p i x) i) ≡ x
updatedₚ {zero}  p fzero x    = refl
updatedₚ {suc n} p fzero x    = refl
updatedₚ {suc n} p (fsuc i) x = updatedₚ (p .snd) i x

updated-neₚ
 : ∀ {n} {ℓ : Fin (suc n) → Level} {P : (i : Fin (suc n)) → Type (ℓ i)}
 → (p : Πᶠ P) (i j : Fin (suc n)) (x : P i)
 → (i ≡ j → ⊥)
 → indexₚ {P = P} (updateₚ {P = P} p i x) j ≡ indexₚ {P = P} p j
updated-neₚ {zero}  p fzero    fzero    x i≠j = absurd (i≠j refl)
updated-neₚ {suc n} p fzero    fzero    x i≠j = absurd (i≠j refl)
updated-neₚ {suc n} p fzero    (fsuc j) x i≠j = refl
updated-neₚ {suc n} p (fsuc i) fzero    x i≠j = refl
updated-neₚ {suc n} p (fsuc i) (fsuc j) x i≠j = updated-neₚ (p .snd) i j x λ p → i≠j (ap fsuc p)
```

# Finitary curried functions

In addition to the finitary dependent products, we can define the type
of _strictly curried functions_ as a more convenient alternative for
non-dependent functions of type $(\Pi^f P) \to X$. Rather than taking
their arguments as tuples, finitary curried functions are... curried.

```agda
Arrᶠ : ∀ {n ℓ ℓ'} (P : (i : Fin n) → Type (ℓ i)) → Type ℓ' → Type (ℓ-maxᶠ ℓ ⊔ ℓ')
Arrᶠ {0} P x     = x
Arrᶠ {suc n} P x = P fzero → Arrᶠ (λ i → P (fsuc i)) x
```

<!--
```agda
∀ᶠ : ∀ n {ℓ ℓ'} (P : (i : Fin n) → Type (ℓ i)) (Q : Πᶠ P → Type ℓ') → Type (ℓ-maxᶠ ℓ ⊔ ℓ')
∀ᶠ zero P Q = Q tt
∀ᶠ (suc n) P Q = (a : P fzero) → ∀ᶠ n (λ i → P (fsuc i)) (λ b → Q (a , b))

apply-∀ᶠ
  : ∀ {n} {ℓ : Fin n → Level} {ℓ'} {P : (i : Fin n) → Type (ℓ i)} {Q : Πᶠ P → Type ℓ'}
  → ∀ᶠ n P Q → (a : Πᶠ P) → Q a
apply-∀ᶠ {zero} f a = f
apply-∀ᶠ {suc n} f (a , as) = apply-∀ᶠ (f a) as

curry-∀ᶠ
  : ∀ {n} {ℓ : Fin n → Level} {ℓ'} {P : (i : Fin n) → Type (ℓ i)} {Q : Πᶠ P → Type ℓ'}
  → ((a : Πᶠ P) → Q a)
  → ∀ᶠ n P Q
curry-∀ᶠ {zero} f = f tt
curry-∀ᶠ {suc n} f a = curry-∀ᶠ {n} λ b → f (a , b)

∀ᶠⁱ : ∀ n {ℓ ℓ'} (P : (i : Fin n) → Type (ℓ i)) (Q : Πᶠ P → Type ℓ') → Type (ℓ-maxᶠ ℓ ⊔ ℓ')
∀ᶠⁱ zero P Q = Q tt
∀ᶠⁱ (suc n) P Q = {a : P fzero} → ∀ᶠⁱ n (λ i → P (fsuc i)) (λ b → Q (a , b))

apply-∀ᶠⁱ
  : ∀ {n} {ℓ : Fin n → Level} {ℓ'} {P : (i : Fin n) → Type (ℓ i)} {Q : Πᶠ P → Type ℓ'}
  → ∀ᶠⁱ n P Q → (a : Πᶠ P) → Q a
apply-∀ᶠⁱ {zero} f a = f
apply-∀ᶠⁱ {suc n} f (a , as) = apply-∀ᶠⁱ (f {a}) as

curry-∀ᶠⁱ
  : ∀ {n} {ℓ : Fin n → Level} {ℓ'} {P : (i : Fin n) → Type (ℓ i)} {Q : Πᶠ P → Type ℓ'}
  → ((a : Πᶠ P) → Q a)
  → ∀ᶠⁱ n P Q
curry-∀ᶠⁱ {zero} f = f tt
curry-∀ᶠⁱ {suc n} f {a} = curry-∀ᶠⁱ {n} λ b → f (a , b)
```
-->

In the generic case, a finitary curried function can be eliminated using
a finitary dependent product; Moreover, curried functions are
"extensional" with respect to this application.

```agda
applyᶠ
  : ∀ {n ℓ'} {ℓ : Fin n → Level} {P : (i : Fin n) → Type (ℓ i)} {X : Type ℓ'}
  → Arrᶠ P X → Πᶠ P → X
applyᶠ {0} f as           = f
applyᶠ {suc n} f (a , as) = applyᶠ (f a) as

funextᶠ
  : ∀ {n ℓ'} {ℓ : Fin n → Level} {P : (i : Fin n) → Type (ℓ i)} {X : Type ℓ'}
  → {f g : Arrᶠ P X} → (∀ (as : Πᶠ P) → applyᶠ f as ≡ applyᶠ g as)
  → f ≡ g
funextᶠ {n = 0}     ps = ps tt
funextᶠ {n = suc n} ps = funext λ x → funextᶠ λ r → ps (x , r)
```

<!--
```agda
Arrᶠ-is-hlevel
  : ∀ {n ℓ'} {ℓ : Fin n → Level} {P : (i : Fin n) → Type (ℓ i)} {X : Type ℓ'}
  → ∀ k → is-hlevel X k → is-hlevel (Arrᶠ P X) k
Arrᶠ-is-hlevel {n = zero}          k hl = hl
Arrᶠ-is-hlevel {n = suc n} {P = P} k hl = fun-is-hlevel k $
  Arrᶠ-is-hlevel {P = λ i → P (fsuc i)} k hl
```
-->

Other than characterising the identity types of finitary curried
functions, we shall need combinators for defining constant functions,
for post-composition with an ordinary function, and for "zipping" two
functions together (applying a binary operation pointwise).

```agda
zipᶠ
  : ∀ {n ℓ' ℓ'' ℓ'''} {ℓ : Fin n → Level} {P : (i : Fin n) → Type (ℓ i)}
      {X : Type ℓ'} {Y : Type ℓ''} {Z : Type ℓ'''}
  → (X → Y → Z) → Arrᶠ P X → Arrᶠ P Y → Arrᶠ P Z
zipᶠ {n = zero}  f g h   = f g h
zipᶠ {n = suc n} f g h a = zipᶠ f (g a) (h a)

mapᶠ
  : ∀ {n ℓ' ℓ''} {ℓ : Fin n → Level} {P : (i : Fin n) → Type (ℓ i)}
      {X : Type ℓ'} {Y : Type ℓ''}
  → (X → Y) → Arrᶠ P X → Arrᶠ P Y
mapᶠ {n = 0}     f g   = f g
mapᶠ {n = suc n} f g x = mapᶠ f (g x)

constᶠ
  : ∀ {n ℓ'} {ℓ : Fin n → Level} {P : (i : Fin n) → Type (ℓ i)} {X : Type ℓ'}
  → X → Arrᶠ P X
constᶠ {n = 0}     x   = x
constᶠ {n = suc n} x _ = constᶠ x
```

Again, when the length is a numeral, these operations are forced to
compute to the expected expressions by the computation of their type.
However, in the generic case, we must express the computation rules
propositionally. Fortunately, simple inductive arguments suffice to
prove them.

```agda
apply-constᶠ
  : ∀ {n ℓ'} {ℓ : Fin n → Level} {P : (i : Fin n) → Type (ℓ i)} {X : Type ℓ'}
    (x : X) (as : Πᶠ P)
  → applyᶠ (constᶠ x) as ≡ x
apply-constᶠ {n = 0}     x as = refl
apply-constᶠ {n = suc n} x as = apply-constᶠ x (as .snd)

apply-zipᶠ
  : ∀ {n ℓ' ℓ'' ℓ'''} {ℓ : Fin n → Level} {P : (i : Fin n) → Type (ℓ i)}
      {X : Type ℓ'} {Y : Type ℓ''} {Z : Type ℓ'''}
    (f : X → Y → Z) (g : Arrᶠ P X) (h : Arrᶠ P Y) (as : Πᶠ P)
  → applyᶠ (zipᶠ f g h) as ≡ f (applyᶠ g as) (applyᶠ h as)
apply-zipᶠ {n = 0}     f g h as = refl
apply-zipᶠ {n = suc n} f g h as = apply-zipᶠ f (g (as .fst)) (h (as .fst)) (as .snd)

apply-mapᶠ
  : ∀ {n ℓ' ℓ''} {ℓ : Fin n → Level} {P : (i : Fin n) → Type (ℓ i)}
      {X : Type ℓ'} {Y : Type ℓ''}
    (f : X → Y) (g : Arrᶠ P X) (as : Πᶠ P)
  → applyᶠ (mapᶠ f g) as ≡ f (applyᶠ g as)
apply-mapᶠ {n = 0}     f g as = refl
apply-mapᶠ {n = suc n} f g as = apply-mapᶠ f (g (as .fst)) (as .snd)
```
