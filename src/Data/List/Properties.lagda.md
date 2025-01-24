<!--
```agda
open import 1Lab.Reflection.HLevel
open import 1Lab.HLevel.Universe
open import 1Lab.HLevel.Closure
open import 1Lab.HLevel
open import 1Lab.Equiv
open import 1Lab.Path
open import 1Lab.Type

open import Data.List.Base
open import Data.Dec.Base
open import Data.Fin.Base hiding (_≤_)
open import Data.Nat.Base
open import Data.Sum.Base
open import Data.Id.Base
open import Data.Bool

open import Meta.Idiom
```
-->

```agda
module Data.List.Properties where
```

# header

<!--
```agda
private variable
  ℓ : Level
  A B : Type ℓ
```
-->

## Path space

Using these lemmas, we can characterise the path space of `List A` in
terms of the path space of `A`. For this, we define by induction a type
family `Code`{.Agda}, which represents paths in `List A` by
iterated products of paths in `A`.

```agda
module ListPath {A : Type ℓ} where
  Code : List A → List A → Type (level-of A)
  Code [] []             = Lift (level-of A) ⊤
  Code [] (x ∷ x₁)       = Lift (level-of A) ⊥
  Code (h ∷ t) []        = Lift (level-of A) ⊥
  Code (h ∷ t) (h' ∷ t') = (h ≡ h') × Code t t'
```

We have a map `encode`{.Agda} which turns a path into a `Code`{.Agda},
and a function `decode`{.Agda} which does the opposite.

```agda
  encode : {xs ys : List A} → xs ≡ ys → Code xs ys
  encode {xs = []} {ys = []} path = lift tt
  encode {xs = []} {ys = x ∷ ys} path = lift (∷≠[] (sym path))
  encode {xs = x ∷ xs} {ys = []} path = lift (∷≠[] path)
  encode {xs = x ∷ xs} {ys = x₁ ∷ ys} path =
    ∷-head-inj path , encode {xs = xs} {ys = ys} (ap tail path)

  decode : {xs ys : List A} → Code xs ys → xs ≡ ys
  decode {[]} {[]} code = refl
  decode {x ∷ xs} {x₁ ∷ ys} (p , q) i = p i ∷ decode q i
```

These maps are inverses by construction:

```agda
  encode-decode : {xs ys : List A} (p : Code xs ys) → encode (decode p) ≡ p
  encode-decode {[]} {[]} (lift tt) = refl
  encode-decode {x ∷ xs} {x₁ ∷ ys} (p , q) i = p , encode-decode q i

  decode-encode : {xs ys : List A} (p : xs ≡ ys) → decode (encode p) ≡ p
  decode-encode = J (λ y p → decode (encode p) ≡ p) de-refl where
    de-refl : {xs : List A} → decode (encode (λ i → xs)) ≡ (λ i → xs)
    de-refl {[]}         = refl
    de-refl {x ∷ xs} i j = x ∷ de-refl {xs = xs} i j

  Code≃Path : {xs ys : List A} → (xs ≡ ys) ≃ Code xs ys
  Code≃Path = Iso→Equiv (encode , iso decode encode-decode decode-encode)
```

Thus we have a characterisation of `Path (List A)` in terms of `Path A`.
We use this to prove that lists preserve h-levels for $n \ge 2$, i.e. if
`A` is a set (or more) then `List A` is a type of the same h-level.

```agda
  List-is-hlevel : (n : Nat) → is-hlevel A (2 + n) → is-hlevel (List A) (2 + n)
  List-is-hlevel n ahl x y = Equiv→is-hlevel (suc n) Code≃Path Code-is-hlevel where
    Code-is-hlevel : {x y : List A} → is-hlevel (Code x y) (suc n)
    Code-is-hlevel {[]} {[]}         = is-prop→is-hlevel-suc λ x y → refl
    Code-is-hlevel {[]} {x ∷ y}      = is-prop→is-hlevel-suc λ x → absurd (lower x)
    Code-is-hlevel {x ∷ x₁} {[]}     = is-prop→is-hlevel-suc λ x → absurd (lower x)
    Code-is-hlevel {x ∷ x₁} {x₂ ∷ y} = ×-is-hlevel (suc n) (ahl _ _) Code-is-hlevel

  instance
    H-Level-List : ∀ {n} ⦃ p : 2 ≤ n ⦄ ⦃ _ : H-Level A n ⦄ → H-Level (List A) n
    H-Level-List {n = suc (suc n)} ⦃ s≤s (s≤s p) ⦄ ⦃ x ⦄ =
      record { has-hlevel = List-is-hlevel n (H-Level.has-hlevel x) }

  is-set→List-is-set : is-set A → is-set (List A)
  is-set→List-is-set = List-is-hlevel zero
```

<!--
```agda
_ : ∀ {ℓ} {A : n-Type ℓ 2} → is-hlevel (List ∣ A ∣) 5
_ = hlevel 5
```
-->

Then we can prove that this operation is associative and has `[]` as
both left and right units:

```agda
++-assoc : ∀ {ℓ} {A : Type ℓ} (xs ys zs : List A)
         → (xs ++ ys) ++ zs ≡ xs ++ (ys ++ zs)
++-assoc [] ys zs = refl
++-assoc (x ∷ xs) ys zs i = x ∷ ++-assoc xs ys zs i

++-idl : ∀ {ℓ} {A : Type ℓ} (xs : List A) → [] ++ xs ≡ xs
++-idl xs i = xs

++-idr : ∀ {ℓ} {A : Type ℓ} (xs : List A) → xs ++ [] ≡ xs
++-idr [] i = []
++-idr (x ∷ xs) i = x ∷ ++-idr xs i
```

## Lemmas

Continuing with the useful lemmas, if the heads and tails of two lists
are identified, then the lists themselves are identified:

```agda
ap-∷ : ∀ {x y : A} {xs ys : List A}
     → x ≡ y → xs ≡ ys
     → Path (List A) (x ∷ xs) (y ∷ ys)
ap-∷ x≡y xs≡ys i = x≡y i ∷ xs≡ys i
```

<!--
```agda
map-id
  : ∀ {ℓ} {A : Type ℓ}
  → (xs : List A)
  → map id xs ≡ xs
map-id [] = refl
map-id (x ∷ xs) = ap (x ∷_) (map-id xs)

map-++
  : ∀ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'}
  → (f : A → B)
  → (xs ys : List A)
  → map f (xs ++ ys) ≡ map f xs ++ map f ys
map-++ f [] ys = refl
map-++ f (x ∷ xs) ys = ap (f x ∷_) (map-++ f xs ys)

take-length : ∀ {ℓ} {A : Type ℓ} (xs : List A) → take (length xs) xs ≡ xs
take-length [] = refl
take-length (x ∷ xs) = ap (x ∷_) (take-length xs)

take-length-more
  : ∀ {ℓ} {A : Type ℓ} (xs : List A) (n : Nat)
  → length xs ≤ n
  → take n xs ≡ xs
take-length-more [] zero wit = refl
take-length-more [] (suc n) wit = refl
take-length-more (x ∷ xs) (suc n) (s≤s wit) = ap (x ∷_) (take-length-more xs n wit)
```
-->

<!--
```agda
all-of-++
  : ∀ {ℓ} {A : Type ℓ}
  → (f : A → Bool)
  → (xs ys : List A)
  → all-of f (xs ++ ys) ≡ and (all-of f xs) (all-of f ys)
all-of-++ f [] ys = refl
all-of-++ f (x ∷ xs) ys =
  ap (and (f x)) (all-of-++ f xs ys)
  ∙ and-associative (f x) (all-of f xs) (all-of f ys)

all-of-map
  : ∀ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'}
  → (f : B → Bool)
  → (g : A → B)
  → (xs : List A)
  → all-of f (map g xs) ≡ all-of (f ∘ g) xs
all-of-map f g [] = refl
all-of-map f g (x ∷ xs) = ap (and (f (g x))) (all-of-map f g xs)

any-of-++
  : ∀ {ℓ} {A : Type ℓ}
  → (f : A → Bool)
  → (xs ys : List A)
  → any-of f (xs ++ ys) ≡ or (any-of f xs) (any-of f ys)
any-of-++ f [] ys = refl
any-of-++ f (x ∷ xs) ys =
  ap (or (f x)) (any-of-++ f xs ys)
  ∙ or-associative (f x) (any-of f xs) (any-of f ys)

any-of-map
  : ∀ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'}
  → (f : B → Bool)
  → (g : A → B)
  → (xs : List A)
  → any-of f (map g xs) ≡ any-of (f ∘ g) xs
any-of-map f g [] = refl
any-of-map f g (x ∷ xs) = ap (or (f (g x))) (any-of-map f g xs)

all-of-or
  : ∀ {ℓ} {A : Type ℓ}
  → (f : A → Bool)
  → (b : Bool) (xs : List A)
  → all-of (λ x → or b (f x)) xs ≡ or b (all-of f xs)
all-of-or f b [] = sym (or-truer b)
all-of-or f b (x ∷ xs) =
  ap (and (or b (f x))) (all-of-or f b xs)
  ∙ sym (or-distrib-andl b (f x) (all-of f xs))

not-all-of
  : ∀ {ℓ} {A : Type ℓ}
  → (f : A → Bool)
  → (xs : List A)
  → not (all-of f xs) ≡ any-of (not ∘ f) xs
not-all-of f [] = refl
not-all-of f (x ∷ xs) =
  not-and≡or-not (f x) (all-of f xs)
  ∙ ap (or (not (f x))) (not-all-of f xs)

not-any-of
  : ∀ {ℓ} {A : Type ℓ}
  → (f : A → Bool)
  → (xs : List A)
  → not (any-of f xs) ≡ all-of (not ∘ f) xs
not-any-of f [] = refl
not-any-of f (x ∷ xs) =
  not-or≡and-not (f x) (any-of f xs)
  ∙ ap (and (not (f x))) (not-any-of f xs)
```
-->

```agda
is-empty : List A → Type
is-empty [] = ⊤
is-empty (_ ∷ _) = ⊥

is-empty? : ∀ (xs : List A) → Dec (is-empty xs)
is-empty? [] = yes tt
is-empty? (x ∷ xs) = no id
```

<!--
```agda
length-tabulate : ∀ {n} (f : Fin n → A) → length (tabulate f) ≡ n
length-tabulate {n = zero}  f = refl
length-tabulate {n = suc n} f = ap suc (length-tabulate (f ∘ fsuc))
```
-->
