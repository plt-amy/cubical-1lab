<!--
```agda
open import 1Lab.Type.Sigma
open import 1Lab.Path
open import 1Lab.Type

open import Data.Product.NAry
open import Data.Maybe.Base
open import Data.Dec.Base
open import Data.Fin.Base
open import Data.Bool

open import Meta.Traversable
open import Meta.Foldable
open import Meta.Append
open import Meta.Idiom
open import Meta.Bind
open import Meta.Alt
```
-->

```agda
module Data.List.Base where
```

# The type of lists {defines=list}

This module contains the definition of the type of lists, and some basic
operations on lists. Properties of these operations are in the module
[`Data.List`], not here.

[`Data.List`]: Data.List.html

<!--
```agda
private variable
  ℓ : Level
  A B : Type ℓ

infixr 20 _∷_
```
-->

```agda
data List {ℓ} (A : Type ℓ) : Type ℓ where
  [] : List A
  _∷_ : A → List A → List A

{-# BUILTIN LIST List #-}
```

One of the first things we set up is list literal syntax (the
`From-product`{.Agda} class) for lists. The list corresponding to an
n-ary product is the list of its elements.

```agda
instance
  From-prod-List : From-product A (λ _ → List A)
  From-prod-List .From-product.from-prod = go where
    go : ∀ n → Vecₓ A n → List A
    go zero xs                = []
    go (suc zero) xs          = xs ∷ []
    go (suc (suc n)) (x , xs) = x ∷ go (suc n) xs

-- Test:
_ : Path (List Nat) [ 1 , 2 , 3 ] (1 ∷ 2 ∷ 3 ∷ [])
_ = refl
```

## “Fields”

```agda
head : A → List A → A
head def []     = def
head _   (x ∷ _) = x

tail : List A → List A
tail []       = []
tail (_ ∷ xs) = xs
```

## Elimination

As with any inductive type, the type of lists-of-As has a
canonically-defined eliminator, but it also has some fairly popular
_recursors_, called "folds":

```agda
List-elim
  : ∀ {ℓ ℓ'} {A : Type ℓ} (P : List A → Type ℓ')
  → P []
  → (∀ x xs → P xs → P (x ∷ xs))
  → ∀ x → P x
List-elim P p[] p∷ []       = p[]
List-elim P p[] p∷ (x ∷ xs) = p∷ x xs (List-elim P p[] p∷ xs)
```

<!--
```agda
instance
  Foldable-List : Foldable (eff List)
  Foldable-List .foldr f x = List-elim _ x (λ x _ → f x)

  Traversable-List : Traversable (eff List)
  Traversable-List = record { traverse = go } where
    go
      : ∀ {M : Effect} ⦃ _ : Idiom M ⦄ (let module M = Effect M) {ℓ ℓ'}
          {a : Type ℓ} {b : Type ℓ'}
      → (a → M.₀ b) → List a → M.₀ (List b)
    go f []       = pure []
    go f (x ∷ xs) = ⦇ f x ∷ go f xs ⦈

foldl : (B → A → B) → B → List A → B
foldl f x []       = x
foldl f x (a ∷ as) = foldl f (f x a) as
```
-->

## Functorial action

It's also possible to lift a function `A → B` to a function `List A →
List B`.

```agda
instance
  Map-List : Map (eff List)
  Map-List = record { map = go } where
    go : (A → B) → List A → List B
    go f []       = []
    go f (x ∷ xs) = f x ∷ go f xs
```

## Monoidal structure

We can define concatenation of lists by recursion:

```agda
_++_ : ∀ {ℓ} {A : Type ℓ} → List A → List A → List A
[]       ++ ys = ys
(x ∷ xs) ++ ys = x ∷ (xs ++ ys)

infixr 8 _++_

instance
  Append-List : ∀ {ℓ} {A : Type ℓ} → Append (List A)
  Append-List = record { mempty = [] ; _<>_ = _++_ }
```

<!--
```agda
map-up : (Nat → A → B) → Nat → List A → List B
map-up f _ []       = []
map-up f n (x ∷ xs) = f n x ∷ map-up f (suc n) xs

length : List A → Nat
length []       = zero
length (x ∷ xs) = suc (length xs)

concat : List (List A) → List A
concat [] = []
concat (x ∷ xs) = x ++ concat xs

count : Nat → List Nat
count zero = []
count (suc n) = 0 ∷ map suc (count n)

product : List Nat → Nat
product [] = 1
product (x ∷ xs) = x * product xs

reverse : List A → List A
reverse = go [] where
  go : List A → List A → List A
  go acc [] = acc
  go acc (x ∷ xs) = go (x ∷ acc) xs

_∷r_ : List A → A → List A
xs ∷r x = xs ++ (x ∷ [])

infixl 20 _∷r_

all=? : (A → A → Bool) → List A → List A → Bool
all=? eq=? [] [] = true
all=? eq=? [] (x ∷ ys) = false
all=? eq=? (x ∷ xs) [] = false
all=? eq=? (x ∷ xs) (y ∷ ys) = and (eq=? x y) (all=? eq=? xs ys)

enumerate : ∀ {ℓ} {A : Type ℓ} → List A → List (Nat × A)
enumerate = go 0 where
  go : Nat → List _ → List (Nat × _)
  go x [] = []
  go x (a ∷ b) = (x , a) ∷ go (suc x) b

take : ∀ {ℓ} {A : Type ℓ} → Nat → List A → List A
take 0       xs       = []
take (suc n) []       = []
take (suc n) (x ∷ xs) = x ∷ take n xs

drop : ∀ {ℓ} {A : Type ℓ} → Nat → List A → List A
drop zero    xs       = xs
drop (suc n) []       = []
drop (suc n) (x ∷ xs) = drop n xs

split-at : ∀ {ℓ} {A : Type ℓ} → Nat → List A → List A × List A
split-at 0       xs       = [] , xs
split-at (suc n) []       = [] , []
split-at (suc n) (x ∷ xs) = ×-map₁ (x ∷_) (split-at n xs)

span : ∀ {ℓ} {A : Type ℓ} (p : A → Bool) → List A → List A × List A
span p [] = [] , []
span p (x ∷ xs) with p x
... | true  = ×-map₁ (x ∷_) (span p xs)
... | false = [] , x ∷ xs

filter : ∀ {ℓ} {A : Type ℓ} (p : A → Bool) → List A → List A
filter p [] = []
filter p (x ∷ xs) with p x
... | true  = x ∷ filter p xs
... | false = filter p xs

intercalate : ∀ {ℓ} {A : Type ℓ} (x : A) (xs : List A) → List A
intercalate x []           = []
intercalate x (y ∷ [])     = y ∷ []
intercalate x (y ∷ z ∷ xs) = y ∷ x ∷ intercalate x (z ∷ xs)

zip : ∀ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'} → List A → List B → List (A × B)
zip [] _ = []
zip _ [] = []
zip (a ∷ as) (b ∷ bs) = (a , b) ∷ zip as bs

unzip : ∀ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'} → List (A × B) → List A × List B
unzip [] = [] , []
unzip ((a , b) ∷ xs) = ×-map (a ∷_) (b ∷_) (unzip xs)

sigma : ∀ {ℓ ℓ'} {A : Type ℓ} {B : A → Type ℓ'} → List A → (∀ a → List (B a)) → List (Σ A B)
sigma [] f = []
sigma (x ∷ xs) f = map (x ,_) (f x) <> sigma xs f

instance
  Idiom-List : Idiom (eff List)
  Idiom-List .pure a = a ∷ []
  Idiom-List ._<*>_ f a = concat ((_<$> a) <$> f)

  Bind-List : Bind (eff List)
  Bind-List ._>>=_ a f = concat (f <$> a)

  Alt-List : Alt (eff List)
  Alt-List .Alt.fail  = []
  Alt-List .Alt._<|>_ = _<>_
```
-->

## Predicates

We can define a function that determines if a boolean-valued
predicate holds on every element of a list.

```agda
all-of : ∀ {ℓ} {A : Type ℓ} → (A → Bool) → List A → Bool
all-of f [] = true
all-of f (x ∷ xs) = and (f x) (all-of f xs)
```

Dually, we can also define a function that determines if a boolean-valued
predicate holds on some element of a list.

```agda
any-of : ∀ {ℓ} {A : Type ℓ} → (A → Bool) → List A → Bool
any-of f [] = false
any-of f (x ∷ xs) = or (f x) (any-of f xs)
```

<!--
```agda
∷-head-inj : ∀ {x y : A} {xs ys} → (x ∷ xs) ≡ (y ∷ ys) → x ≡ y
∷-head-inj {x = x} p = ap (head x) p

∷-tail-inj : ∀ {x y : A} {xs ys} → (x ∷ xs) ≡ (y ∷ ys) → xs ≡ ys
∷-tail-inj p = ap tail p

∷≠[] : ∀ {x : A} {xs} → ¬ (x ∷ xs) ≡ []
∷≠[] {A = A} p = subst distinguish p tt where
  distinguish : List A → Type
  distinguish []     = ⊥
  distinguish (_ ∷ _) = ⊤

instance
  Discrete-List : ∀ ⦃ d : Discrete A ⦄ → Discrete (List A)
  Discrete-List {x = []}     {y = []}     = yes refl
  Discrete-List {x = []}     {y = x ∷ y}  = no λ p → ∷≠[] (sym p)
  Discrete-List {x = x ∷ xs} {y = []}     = no ∷≠[]
  Discrete-List {x = x ∷ xs} {y = y ∷ ys} = case x ≡? y of λ where
    (yes x=y) → case Discrete-List {x = xs} {ys} of λ where
      (yes xs=ys) → yes (ap₂ _∷_ x=y xs=ys)
      (no  xs≠ys) → no λ p → xs≠ys (∷-tail-inj p)
    (no x≠y)      → no λ p → x≠y (∷-head-inj p)

traverse-up
  : ∀ {M : Effect} ⦃ _ : Idiom M ⦄ (let module M = Effect M) {ℓ ℓ'}
    {a : Type ℓ} {b : Type ℓ'}
  → (Nat → a → M.₀ b) → Nat → List a → M.₀ (List b)
traverse-up f n xs = sequence (map-up f n xs)

lookup : ⦃ _ : Discrete A ⦄ → A → List (A × B) → Maybe B
lookup x [] = nothing
lookup x ((k , v) ∷ xs) with x ≡? k
... | yes _ = just v
... | no  _ = lookup x xs

_!_ : (l : List A) → Fin (length l) → A
(x ∷ xs) ! n with fin-view n
... | zero  = x
... | suc i = xs ! i

tabulate : ∀ {n} (f : Fin n → A) → List A
tabulate {n = zero}  f = []
tabulate {n = suc n} f = f fzero ∷ tabulate (f ∘ fsuc)
```
-->
