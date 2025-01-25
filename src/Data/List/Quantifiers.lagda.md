<!--
```agda
open import 1Lab.Prelude

open import Data.List.Membership
open import Data.List.Base
open import Data.Dec.Base
open import Data.Sum.Base
```
-->

```agda
module Data.List.Quantifiers where
```

# Quantifiers over lists

```agda
data Some {ℓ ℓ'} {A : Type ℓ} (P : A → Type ℓ') : List A → Type (ℓ ⊔ ℓ') where
  here  : ∀ {x xs} → P x       → Some P (x ∷ xs)
  there : ∀ {x xs} → Some P xs → Some P (x ∷ xs)

data All {ℓ ℓ'} {A : Type ℓ} (P : A → Type ℓ') : List A → Type (ℓ ⊔ ℓ') where
  []  :                             All P []
  _∷_ : ∀ {x xs} → P x → All P xs → All P (x ∷ xs)
```

<!--
```agda
private variable
  ℓ ℓ' : Level
  A B : Type ℓ
  P Q : A → Type ℓ'
  x y : A
  xs ys : List A
```
-->

```agda
Some-elim
  : ∀ {P : A → Type ℓ} (R : (xs : List A) → Some P xs → Type ℓ')
  → (∀ {x} {xs} → (p : P x) → R (x ∷ xs) (here p))
  → (∀ {x} {xs} → (p : Some P xs) → R xs p → R (x ∷ xs) (there p))
  → ∀ {xs : List A} (s : Some P xs) → R xs s
Some-elim R h t (here p) = h p
Some-elim R h t (there p) = t p (Some-elim R h t p)

Some-rec
  : ∀ {P : A → Type ℓ} {X : Type ℓ'}
  → (∀ {x} → P x → X)
  → (X → X)
  → ∀ {xs : List A} → Some P xs → X
Some-rec h t (here p) = h p
Some-rec h t (there p) = t (Some-rec h t p)

All-elim
  : ∀ {P : A → Type ℓ} (R : (xs : List A) → All P xs → Type ℓ')
  → R [] []
  → ( ∀ {x} {xs} (px : P x) (pxs : All P xs)
    → R xs pxs
    → R (x ∷ xs) (px ∷ pxs))
  → ∀ {xs : List A} (a : All P xs) → R xs a
All-elim R n e [] = n
All-elim R n e (x ∷ a) = e x a (All-elim R n e a)

All-rec
  : ∀ {P : A → Type ℓ} {X : Type ℓ'}
  → X
  → (∀ {x} → P x → X → X)
  → ∀ {xs : List A} → All P xs → X
All-rec n e [] = n
All-rec n e (p ∷ a) = e p (e p n)
```

```agda
∷-all-head : All P (x ∷ xs) → P x
∷-all-head (px ∷ _) = px

∷-all-tail : All P (x ∷ xs) → All P xs
∷-all-tail (_ ∷ pxs) = pxs

∷-all-× : All P (x ∷ xs) → P x × All P xs
∷-all-× (px ∷ pxs) = px , pxs

¬some-[] : ¬ Some P []
¬some-[] ()

∷-some-⊎ : Some P (x ∷ xs) → P x ⊎ Some P xs
∷-some-⊎ (here px) = inl px
∷-some-⊎ (there pxs) = inr pxs

∷-¬some-head : ¬ (Some P (x ∷ xs)) → ¬ (P x)
∷-¬some-head ¬some px = ¬some (here px)

∷-¬some-tail : ¬ (Some P (x ∷ xs)) → ¬ (Some P xs)
∷-¬some-tail ¬some px = ¬some (there px)
```

```agda
some-map : (∀ {x} → P x → Q x) → Some P xs → Some Q xs
some-map f (here px) = here (f px)
some-map f (there pxs) = there (some-map f pxs)

all-map : (∀ {x} → P x → Q x) → All P xs → All Q xs
all-map f [] = []
all-map f (px ∷ pxs) = f px ∷ all-map f pxs

not-some→all-not
  : ¬ (Some P xs)
  → All (λ x → ¬ (P x)) xs
not-some→all-not {xs = []} ¬some = []
not-some→all-not {xs = x ∷ xs} ¬some =
  ∷-¬some-head ¬some ∷ not-some→all-not (∷-¬some-tail ¬some)

all-not→not-some
  : All (λ x → ¬ P x) xs
  → ¬ (Some P xs)
all-not→not-some (¬P ∷ ps) (here p)  = ¬P p
all-not→not-some (¬P ∷ ps) (there x) = all-not→not-some ps x
```

```agda
some?
  : ∀ {P : A → Type ℓ}
  → (∀ x → Dec (P x))
  → (xs : List A) → Dec (Some P xs)
some? P? [] = no ¬some-[]
some? P? (x ∷ xs) with P? x
... | yes px = yes (here px)
... | no ¬px with some? P? xs
... | yes pxs = yes (there pxs)
... | no ¬pxs = no ([ ¬px , ¬pxs ] ∘ ∷-some-⊎)

all?
  : ∀ {P : A → Type ℓ}
  → (∀ x → Dec (P x))
  → (xs : List A) → Dec (All P xs)
all? P? [] = yes []
all? P? (x ∷ xs) with P? x | all? P? xs
... | yes p | yes ps = yes (p ∷ ps)
... | yes p | no ¬ps = no (¬ps ∘ ∷-all-tail)
... | no ¬p | q = no (¬p ∘ ∷-all-head)
```

<!--
```agda
instance
  Dec-Some : ∀ {P : A → Type ℓ} → ⦃ ∀ {x} → Dec (P x) ⦄ → {xs : List A} → Dec (Some P xs)
  Dec-Some {P = P} = some? (λ x → holds? (P x)) _

  Dec-All : ∀ {P : A → Type ℓ} → ⦃ ∀ {x} → Dec (P x) ⦄ → {xs : List A} → Dec (All P xs)
  Dec-All {P = P} = all? (λ x → holds? (P x)) _
```
-->


```agda
all-∈ : All P xs → x ∈ xs → P x
all-∈ {P = P} (px ∷ pxs) (here p) = substᵢ P (symᵢ p) px
all-∈ (px ∷ pxs) (there x∈xs) = all-∈ pxs x∈xs

to-all : (∀ x → x ∈ xs → P x) → All P xs
to-all {xs = []} p = []
to-all {xs = x ∷ xs} p = p x (here reflᵢ) ∷ to-all (λ i w → p i (there w))
```

<!--
```agda
instance
  H-Level-All : ∀ {ℓ ℓ'} {A : Type ℓ} {P : A → Type ℓ'} {n} ⦃ _ : ∀ {x} → H-Level (P x) n ⦄ {xs} → H-Level (All P xs) n
  H-Level-All {P = P} {xs = xs} = hlevel-instance
    (retract→is-hlevel {A = ∀ x → x ∈ₗ xs → P x} _ to-all (λ a _ → all-∈ a) inv (hlevel _))
    where
      inv : ∀ {xs} → is-left-inverse (to-all {xs = xs} {P = P}) (λ a z → all-∈ a)
      inv {xs} [] = refl
      inv {x ∷ xs} (y ∷ ys) i = y ∷ inv ys i
```
-->
