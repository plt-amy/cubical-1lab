<!--
```agda
open import 1Lab.Prelude

open import Data.List.Properties
open import Data.List.Base
open import Data.Dec.Base
open import Data.Fin.Base
open import Data.Sum.Base
open import Data.Id.Base
open import Data.Bool

open import Meta.Idiom
```
-->

```agda
module Data.List.Membership where
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

# Membership in lists

We can inductively define a notion of membership between elements $a :
A$ and lists $as : \List{A}$; we write $a \in_l as$. However, note that
this notion of membership is **not** a [[proposition]]! The type $a
\in_l as$ has at least as many inhabitants as there are occurrences of
$a$ in $as$; and if the type $A$ is not a set, then each proof of $a \in
as$ can be acted on by a loop on $a$ to give a new proof.

```agda
data _∈ₗ_ {ℓ} {A : Type ℓ} (x : A) : List A → Type ℓ where
  here  : ∀ {x'} (p : x ≡ x') → x ∈ₗ (x' ∷ xs)
  there : (p : x ∈ₗ xs)       → x ∈ₗ (y ∷ xs)
```

<!--
```agda
instance
  Membership-List : ∀ {ℓ} {A : Type ℓ} → Membership A (List A) ℓ
  Membership-List = record { _∈_ = _∈ₗ_ }
```
-->

There is a more (homotopically) straightforward characterisation of
membership in lists: the [[fibres]] of the lookup function `xs !
i`{.Agda ident=!_}. These are given by an index $i :
[\operatorname{length}(xs)]$, living in the [[standard finite set]] with
as many elements as the list has positions, together with a proof that
$xs \mathbin{!} i = x$.

The inverse equivalences between these notions is defined below; the
proof that they are _are_ inverses is a straightforward induction in
both cases, so it's omitted for space.

```agda
element→!-fibre : ∀ {x : A} {xs} → x ∈ xs → fibre (xs !_) x
element→!-fibre (here p) = fzero , sym p
element→!-fibre (there prf) with element→!-fibre prf
... | ix , p = fsuc ix , p

!-fibre→element : ∀ {x : A} {xs} → fibre (xs !_) x → x ∈ xs
!-fibre→element {A = A} {x = x} = λ (ix , p) → go ix p module !-fibre→element where
  go : ∀ {xs} (ix : Fin (length xs)) → xs ! ix ≡ x → x ∈ xs
  go ix _  with fin-view ix
  go {xs = x ∷ xs} _ p | zero     = here  (sym p)
  go {xs = x ∷ xs} _ p | (suc ix) = there (go ix p)
```

The equivalence between these definitions explains why $a \in_l as$ can
be so complicated. First, it's at least a set, since it stores the
index. Second, it stores a path, which can be arbitrarily complicated
depending on the type $A$.

<!--
```agda
!-fibre→element→fibre : ∀ {x : A} {xs} (f : fibre (xs !_) x) → element→!-fibre (!-fibre→element f) ≡ f
!-fibre→element→fibre {A = A} {x = x} (ix , p) = go ix p where
  go : ∀ {xs} (ix : Fin (length xs)) (p : xs ! ix ≡ x) → element→!-fibre (!-fibre→element.go {xs = xs} ix p) ≡ (ix , p)
  go ix p with fin-view ix
  go {xs = x ∷ xs} _ p | zero = refl
  go {xs = x ∷ xs} _ p | suc ix = Σ-pathp (ap fsuc (ap fst p')) (ap snd p')
    where p' = go {xs = xs} ix p

element→!-fibre→element
  : {x : A} {xs : List A} (p : x ∈ xs) → p ≡ !-fibre→element (element→!-fibre p)
element→!-fibre→element (here p)  = refl
element→!-fibre→element (there p) = ap there (element→!-fibre→element p)

element≃!-fibre : ∀ {x : A} {xs} → (x ∈ₗ xs) ≃ fibre (xs !_) x
element≃!-fibre .fst = element→!-fibre
element≃!-fibre .snd = is-iso→is-equiv λ where
  .is-iso.inv  p → !-fibre→element p
  .is-iso.rinv p → !-fibre→element→fibre p
  .is-iso.linv p → sym (element→!-fibre→element p)
```
-->

Despite the potential complexity of $a \in_l as$, we do note that if $A$
is a [[set]], then all that matters is the index; If $A$ is moreover
[[discrete]], then $a \in_l as$ is [[decidable]].

```agda
elem? : ⦃ _ : Discrete A ⦄ (x : A) (xs : List A) → Dec (x ∈ xs)
elem? x [] = no λ ()
elem? x (y ∷ xs) with x ≡ᵢ? y
... | yes reflᵢ = yes (here refl)
... | no ¬p with elem? x xs
... | yes p = yes (there p)
... | no ¬q = no λ { (here p) → ¬p (Id≃path.from p) ; (there q) → ¬q q }
```

<!--
```agda
instance
  Dec-∈ₗ : ⦃ _ : Discrete A ⦄ {x : A} {xs : List A} → Dec (x ∈ xs)
  Dec-∈ₗ {x = x} {xs} = elem? x xs
```
-->

## Removing duplicates

Still working with a discrete type, we can define a function
`nub`{.Agda} which removes duplicates from a list. It is constructed by
inductively deciding whether or not to keep the head of the list,
discarding those which already appear further down. This has terrible
the terrible time complexity $O(n^2)$, but it works for an arbitrary
discrete type, which is the best possible generality.

```agda
nub-cons : (x : A) (xs : List A) → Dec (x ∈ xs) → List A
nub-cons x xs (yes _) = xs
nub-cons x xs (no _)  = x ∷ xs

nub : ⦃ _ : Discrete A ⦄ → List A → List A
nub []       = []
nub (x ∷ xs) = nub-cons x (nub xs) (elem? x (nub xs))
```

The function `nub`{.Agda} is characterised by the following two facts.
First, membership in `nub`{.Agda} is a proposition --- each element
appears at most once. Second, membership is (logically) preserved when
`nub`{.Agda}bing a list --- note that the function mapping old indices
to new indices is not injective, since all occurrences of an element
will be mapped to the same (first) occurrence in the deduplicated list.

```agda
member-nub-is-prop
  : ∀ ⦃ _ : Discrete A ⦄ {x : A} (xs : List A) → is-prop (x ∈ nub xs)
member→member-nub
  : ∀ ⦃ _ : Discrete A ⦄ {x : A} {xs : List A} → x ∈ xs → x ∈ nub xs
```

<details>
<summary>The proofs here are also straightforward inductive arguments.</summary>

```agda
member-nub-is-prop (x ∷ xs) p1 p2 with elem? x (nub xs) | p1 | p2
... | yes p | p1 | p2 = member-nub-is-prop xs p1 p2
... | no ¬p | here  p1 | here  p2 = ap _∈ₗ_.here (Discrete→is-set auto _ _ p1 p2)
... | no ¬p | here  p1 | there p2 = absurd (¬p (subst (_∈ nub xs) p1 p2))
... | no ¬p | there p1 | here  p2 = absurd (¬p (subst (_∈ nub xs) p2 p1))
... | no ¬p | there p1 | there p2 = ap there (member-nub-is-prop xs p1 p2)

member→member-nub {xs = x ∷ xs} (here p) with elem? x (nub xs)
... | yes x∈nub = subst (_∈ nub xs) (sym p) x∈nub
... | no ¬x∈nub = here p
member→member-nub {xs = x ∷ xs} (there α) with elem? x (nub xs)
... | yes x∈nub = member→member-nub α
... | no ¬x∈nub = there (member→member-nub α)
```

</details>

<!--
```agda
!-tabulate : ∀ {n} (f : Fin n → A) i → tabulate f ! i ≡ f (subst Fin (length-tabulate f) i)
!-tabulate _ ix with fin-view ix
!-tabulate {n = suc n} f _ | zero  = refl
!-tabulate {n = suc n} f _ | suc i = !-tabulate (f ∘ fsuc) i

!-tabulate-fibre : ∀ {n} (f : Fin n → A) x → fibre (tabulate f !_) x ≃ fibre f x
!-tabulate-fibre f x = Σ-ap (path→equiv (ap Fin (length-tabulate f))) λ i →
  path→equiv (ap (_≡ x) (!-tabulate f i))

member-tabulate : ∀ {n} (f : Fin n → A) x → (x ∈ tabulate f) ≃ fibre f x
member-tabulate f x = element≃!-fibre ∙e !-tabulate-fibre f x
```
-->

<!--
```agda
map-member
  : ∀ {A : Type ℓ} {B : Type ℓ'} (f : A → B) {x : A} {xs : List A}
  → x ∈ xs → f x ∈ map f xs
map-member f (here p)  = here (ap f p)
map-member f (there x) = there (map-member f x)

module _ {ℓ ℓ'} {A : Type ℓ} {B : Type ℓ'} (f : A → B) where
  member-map→fibre : ∀ {y} xs → y ∈ₗ map f xs → Σ[ f ∈ fibre f y ] (f .fst ∈ₗ xs)
  member-map→fibre (x ∷ xs) (here p)  = (x , sym p) , here refl
  member-map→fibre (x ∷ xs) (there p) =
    let (f , ix) = member-map→fibre xs p
      in f , there ix

  fibre→member-map : ∀ {y} xs (fb : fibre f y) → fb .fst ∈ₗ xs → y ∈ₗ map f xs
  fibre→member-map (_ ∷ xs) (x , p) (here q)  = here (sym p ∙ ap f q)
  fibre→member-map (_ ∷ xs) (x , p) (there q) = there (fibre→member-map xs (x , p) q)

  member-map→fibre→member
    : ∀ {y} xs (p : y ∈ₗ map f xs)
    → fibre→member-map xs (member-map→fibre xs p .fst) (member-map→fibre xs p .snd) ≡ p
  member-map→fibre→member (x ∷ xs) (here p)  = ap here (∙-idr _)
  member-map→fibre→member (x ∷ xs) (there p) = ap there (member-map→fibre→member xs p)

++-memberₗ : x ∈ₗ xs → x ∈ₗ (xs ++ ys)
++-memberₗ (here p)  = here p
++-memberₗ (there p) = there (++-memberₗ p)

++-memberᵣ : x ∈ₗ ys → x ∈ₗ (xs ++ ys)
++-memberᵣ {xs = []}     p = p
++-memberᵣ {xs = x ∷ xs} p = there (++-memberᵣ p)

Member-++-view
  : ∀ {ℓ} {A : Type ℓ} (x : A) (xs : List A) (ys : List A)
  → (p : x ∈ₗ (xs ++ ys)) → Type _
Member-++-view x xs ys p = (Σ[ q ∈ x ∈ₗ xs ] (++-memberₗ q ≡ p)) ⊎ (Σ[ q ∈ x ∈ₗ ys ] (++-memberᵣ q ≡ p))

member-++-view
  : ∀ {ℓ} {A : Type ℓ} {x : A} (xs : List A) {ys : List A}
  → (p : x ∈ₗ (xs ++ ys)) → Member-++-view x xs ys p
member-++-view []       p         = inr (p , refl)
member-++-view (x ∷ xs) (here p)  = inl (here p , refl)
member-++-view (x ∷ xs) (there p) with member-++-view xs p
... | inl (p , q) = inl (there p , ap there q)
... | inr (p , q) = inr (p , ap there q)
```
-->

<!--
```agda
any-one-of
  : ∀ {ℓ} {A : Type ℓ}
  → (f : A → Bool) (x : A) (xs : List A)
  → x ∈ xs → f x ≡ true
  → any-of f xs ≡ true
any-one-of f x (y ∷ xs) (here x=y) x-true =
  ap₂ or (subst (λ e → f e ≡ true) x=y x-true) refl
any-one-of f x (y ∷ xs) (there x∈xs) x-true =
  ap₂ or refl (any-one-of f x xs x∈xs x-true) ∙ or-truer _
```
-->
