open import 1Lab.Prelude

open import Data.List.Membership
open import Data.Finset.Base
open import Data.Sum.Base
open import Data.Nat.Base
open import Data.Dec

import 1Lab.Reflection

open 1Lab.Reflection using (Idiom-TC ; lit ; nat ; con₀ ; List ; [] ; _∷_)

module Data.Finset.Properties where

private variable
  ℓ ℓ' ℓ'' : Level
  A B C : Type ℓ

map-∈ᶠˢ : (f : A → B) {x : B} (xs : Finset A) → x ∈ mapᶠˢ f xs → ∃[ (y , _) ∈ fibreᵢ f x ] y ∈ xs
map-∈ᶠˢ f {x} xs w = Finset-elim-prop (λ xs → x ∈ mapᶠˢ f xs → ∃[ (y , _) ∈ fibreᵢ f x ] y ∈ xs)
  (λ w → absurd (¬mem-[] w))
  (λ y ind → ∈ᶠˢ-split (λ { p → inc ((y , symᵢ p) , hereₛ) }) λ w → case ind w of λ x p h → inc ((x , p) , thereₛ h))
  xs w

∈ᶠˢ-map : ∀ (f : A → B) {x} (xs : Finset A) → x ∈ᶠˢ xs → f x ∈ᶠˢ mapᶠˢ f xs
∈ᶠˢ-map f {x} = ∈ᶠˢ-elim (λ xs _ → f x ∈ᶠˢ map f xs) hereₛ (λ q → thereₛ)

∈ᶠˢ-filter : ∀ {P : A → Type ℓ} ⦃ d : ∀ {x} → Dec (P x) ⦄ {x : A} xs → x ∈ᶠˢ xs → P x → x ∈ᶠˢ filter P xs
∈ᶠˢ-filter {P = P} ⦃ d ⦄ {x} xs mem px = ∈ᶠˢ-elim (λ xs _ → x ∈ᶠˢ filter P xs)
  (λ {xs} → case d {x} return (λ p → x ∈ᶠˢ cons-if p x (filter P xs)) of λ { (yes _) → hereₛ ; (no ¬px) → absurd (¬px px)})
  (λ {y} {xs} q ind → case d {y} return (λ p → x ∈ᶠˢ cons-if p y (filter P xs)) of λ { (yes _) → thereₛ ind ; (no ¬px) → ind })
  xs mem

filter-∈ᶠˢ : ∀ {P : A → Type ℓ} ⦃ d : ∀ {x} → Dec (P x) ⦄ {x : A} xs → x ∈ᶠˢ filter P xs → ∥ P x ∥
filter-∈ᶠˢ {P = P} ⦃ d = d ⦄ {x = x} = Finset-elim-prop (λ xs → x ∈ filter P xs → ∥ P x ∥)
  (λ w → absurd (¬mem-[] w))
  (λ y {xs} ind → case d {y} return (λ p → x ∈ᶠˢ cons-if p y (filter P xs) → ∥ P x ∥) of λ where
    (yes py) → ∈ᶠˢ-split (λ p → inc (substᵢ P (symᵢ p) py)) ind
    (no ¬py) q → ind q)

uncons : (x : A) (xs : Finset A) → x ∈ᶠˢ xs → xs ≡ x ∷ xs
uncons x = Finset-elim-prop _ (λ x → absurd (¬mem-[] x)) λ y {xs} ih → ∈ᶠˢ-split
  (λ { reflᵢ → sym (∷-dup _ _) })
  (λ w → ap (y ∷_) (ih w) ∙ ∷-swap _ _ _)

∈ᶠˢ-unionl : (x : A) (xs ys : Finset A) → x ∈ᶠˢ xs → x ∈ᶠˢ (xs <> ys)
∈ᶠˢ-unionl x xs ys p = ∈ᶠˢ-elim (λ xs _ → x ∈ᶠˢ (xs <> ys)) hereₛ (λ q x → thereₛ x) xs p

∈ᶠˢ-unionr : (x : A) (xs ys : Finset A) → x ∈ᶠˢ ys → x ∈ᶠˢ (xs <> ys)
∈ᶠˢ-unionr x xs ys p = Finset-elim-prop (λ xs → x ∈ᶠˢ (xs <> ys)) p (λ x p → thereₛ p) xs

∈ᶠˢ-union : (x : A) (xs ys : Finset A) → x ∈ᶠˢ (xs <> ys) → ∥ (x ∈ᶠˢ xs) ⊎ (x ∈ᶠˢ ys) ∥
∈ᶠˢ-union x = Finset-elim-prop _ (λ ys w → inc (inr w)) λ x {xs} ind ys →
  ∈ᶠˢ-split (λ p → inc (inl (hereₛ' p)))
    λ w → case ind ys w of λ where
      (inl h) → inc (inl (thereₛ h))
      (inr t) → inc (inr t)

sigma-∈ᶠˢ
  : ∀ {ℓ ℓ'} {A : Type ℓ} {P : A → Type ℓ'} {x y}
  → ∀ (f : ∀ x → Finset (P x)) xs
  → x ∈ᶠˢ xs → y ∈ᶠˢ f x → (x , y) ∈ᶠˢ sigma xs f
sigma-∈ᶠˢ {P = P} {x = x₀} {y = y₀} f xs mx my = ∈ᶠˢ-elim (λ xs _ → (y : P x₀) → y ∈ f x₀ → (x₀ , y) ∈ sigma xs f)
  (λ {xs} y ym → ∈ᶠˢ-unionl (x₀ , y) (map (x₀ ,_) (f x₀)) (sigma xs f) (∈ᶠˢ-map (x₀ ,_) (f x₀) ym))
  (λ {y} {xs} q ind py ym → ∈ᶠˢ-unionr (x₀ , py) (map (y ,_) (f y)) (sigma xs f) (ind py ym))
  xs mx y₀ my

∈ᶠˢ-sigma
  : ∀ {ℓ ℓ'} {A : Type ℓ} {P : A → Type ℓ'} {x y} (f : ∀ x → Finset (P x)) xs
  → (x , y) ∈ᶠˢ sigma xs f → x ∈ xs × y ∈ f x
∈ᶠˢ-sigma {P = P} {x = x} {y} f = Finset-elim-prop (λ xs → (x , y) ∈ᶠˢ sigma xs f → x ∈ xs × y ∈ f x)
  (λ w → absurd (¬mem-[] w))
  (λ x' {xs} ind w → case ∈ᶠˢ-union (x , y) (map (x' ,_) (f x')) (sigma xs f) w of λ where
    (inl w) → case map-∈ᶠˢ (x' ,_) (f x') w of λ px' pf mem →
      let (p , q) = id-Σ (symᵢ pf) in
      Jᵢ (λ x' p → ∀ px' (q : Id-over P p y px') → px' ∈ᶠˢ f x' → (x ∈ᶠˢ (x' ∷ xs)) × (y ∈ᶠˢ f x))
         (λ px'' → Jᵢ (λ px'' _ → px'' ∈ f x → (x ∈ (x ∷ xs)) × (y ∈ f x)) (hereₛ ,_))
         p px' q mem
    (inr w) → case ind w of λ h1 h2 → thereₛ h1 , h2)

intersection-∈ᶠˢ : ⦃ _ : Discrete A ⦄ (x : A) (xs ys : Finset A) → x ∈ xs → x ∈ ys → x ∈ intersection xs ys
intersection-∈ᶠˢ x xs ys mxs mys = ∈ᶠˢ-elim (λ xs _ → x ∈ intersection xs ys)
  (λ {xs} → caseᵈ x ∈ ys return (λ p → x ∈ cons-if p x (intersection xs ys)) of λ where
    (yes _)   → hereₛ
    (no ¬mys) → absurd (¬mys mys))
  (λ {y} {xs} w q → there-cons-if (holds? _) y x (intersection xs ys) q)
  xs mxs

∈ᶠˢ-intersection : ⦃ d : Discrete A ⦄ (x : A) (xs ys : Finset A) → x ∈ intersection xs ys → (x ∈ xs) × (x ∈ ys)
∈ᶠˢ-intersection x xs ys w = Finset-elim-prop (λ xs → x ∈ intersection xs ys → x ∈ xs × x ∈ ys)
  (λ w → absurd (¬mem-[] w))
  (λ y {tl} ind → caseᵈ (y ∈ ys) return (λ p → x ∈ cons-if p y (intersection tl ys) → x ∈ (y ∷ tl) × x ∈ ys) of λ where
    (yes q) → ∈ᶠˢ-split (λ p → substᵢ (λ e → x ∈ᶠˢ (e ∷ tl)) p hereₛ , substᵢ (_∈ᶠˢ ys) (symᵢ p) q) λ w → case ind w of λ a b → thereₛ a , b
    (no ¬p) mem → case ind mem of λ a b → thereₛ a , b)
  xs w

from-list-∈ᶠˢ : ∀ {x : A} {xs} → x ∈ᶠˢ from-list xs → ∥ x ∈ₗ xs ∥
from-list-∈ᶠˢ {xs = []} p = absurd (¬mem-[] p)
from-list-∈ᶠˢ {xs = x ∷ xs} = ∈ᶠˢ-split (λ p → inc (here p)) (λ w → there <$> from-list-∈ᶠˢ w)

∈ᶠˢ-from-list : ∀ {x : A} {xs} → x ∈ₗ xs → x ∈ᶠˢ from-list xs
∈ᶠˢ-from-list (here p)  = hereₛ' p
∈ᶠˢ-from-list (there x) = thereₛ (∈ᶠˢ-from-list x)

instance
  Dec-All
    : ∀ {P : A → Type ℓ} ⦃ _ : ∀ {x} → H-Level (P x) 1 ⦄ ⦃ d : ∀ {x} → Dec (P x) ⦄ {xs : Finset A}
    → Dec (All P xs)
  Dec-All {P = P} {xs} = Finset-elim-prop (λ xs → Dec (All P xs)) (yes all-nil)
    (λ { x (yes xs) → caseᵈ P x of λ { (yes x) → yes (all-cons x xs) ; (no ¬x) → no (λ xxs → ¬x (from-all _ xxs hereₛ) )}
       ; _ (no ¬xs) → no (λ xxs → ¬xs (to-all _ (λ x w → from-all _ xxs (thereₛ w))))
       })
    xs

abstract
  subset→is-union : (xs ys : Finset A) → xs ⊆ ys → ys ≡ (xs <> ys)
  subset→is-union = Finset-elim-prop _ (λ ys p → refl) λ x {xs} ih ys sube →
    ys             ≡⟨ uncons x ys (sube x hereₛ) ⟩
    x ∷ ys         ≡⟨ ap (x ∷_) (ih ys λ x x∈xs → sube x (thereₛ x∈xs)) ⟩
    x ∷ (xs <> ys) ∎

  finset-ext : {xs ys : Finset A} → xs ⊆ ys → ys ⊆ xs → xs ≡ ys
  finset-ext {xs = xs} {ys} p1 p2 =
    let
      p : xs ≡ (ys <> xs)
      p = subset→is-union ys xs p2

      q : ys ≡ (xs <> ys)
      q = subset→is-union xs ys p1
    in p ·· union-comm ys xs ·· sym q

instance
  Discrete-Finset : ⦃ _ : Discrete A ⦄ → Discrete (Finset A)
  Discrete-Finset {x = xs} {ys} = case holds? (All (_∈ᶠˢ ys) xs × All (_∈ᶠˢ xs) ys) of λ where
    (yes (s1 , s2)) → yes $ finset-ext (λ a → from-all _ s1) (λ a → from-all _ s2)
    (no ¬sub)       → no λ p → ¬sub (to-all xs (λ a → subst (a ∈_) p) , to-all ys (λ a → subst (a ∈_) (sym p)))

-- record Finite {ℓ} (A : Type ℓ) : Type ℓ where
--   field
--     univ : Finset A
--     has : (x : A) → x ∈ univ

-- open Finite

-- instance
--   H-Level-Finite : ∀ {n} → H-Level (Finite A) (suc n)
--   H-Level-Finite = prop-instance {T = Finite _} λ where
--     x y i .univ  → finset-ext (λ a _ → y .has a) (λ a _ → x .has a) i
--     x y i .has a → is-prop→pathp (λ i → hlevel {T = a ∈ᶠˢ finset-ext (λ a _ → y .has a) (λ a _ → x .has a) i} 1) (x .has a) (y .has a) i

-- open import Data.Fin.Base

-- test : Finite A → ∃[ n ∈ Nat ] (Fin n ↠ A)
-- test record { univ = xs ; has = h } = do
--   (xs , p) ← from-list-surj xs

--   let
--     surj : is-surjective (xs !_)
--     surj a = do
--       (ix , p) ← member→lookup <$> from-list-mem {xs = xs} (subst (a ∈ᶠˢ_) (sym p) (h a))
--       pure (ix , Id≃path.to p)

--   pure (length xs , (xs !_) , surj)

-- instance
--   Finite-⊤ : Finite ⊤
--   Finite-⊤ = record { univ = pure tt ; has = λ x → hereₛ }

--   Finite-⊥ : Finite ⊥
--   Finite-⊥ = record { univ = [] ; has = λ () }

--   Finite-⊎ : ⦃ _ : Finite A ⦄ ⦃ _ : Finite B ⦄ → Finite (A ⊎ B)
--   Finite-⊎ ⦃ a ⦄ ⦃ b ⦄ .univ = map inl (a .univ) <> map inr (b .univ)
--   Finite-⊎ ⦃ a ⦄ ⦃ b ⦄ .has (inl x) = ∈ᶠˢ-unionl (inl x) (map inl (a .univ)) _ (∈ᶠˢ-map inl (a .univ) (a .has x))
--   Finite-⊎ ⦃ a ⦄ ⦃ b ⦄ .has (inr x) = ∈ᶠˢ-unionr (inr x) (map inl (a .univ)) _ (∈ᶠˢ-map inr (b .univ) (b .has x))

--   Finite-Fin : ∀ {n} → Finite (Fin n)
--   Finite-Fin {n} = record { univ = n-univ n ; has = n-has n } where
--     n-univ : ∀ n → Finset (Fin n)
--     n-univ zero    = []
--     n-univ (suc n) = fzero ∷ map fsuc (n-univ n)

--     n-has : ∀ n (x : Fin n) → x ∈ n-univ n
--     n-has _ f with fin-view f
--     ... | zero  = hereₛ
--     ... | suc x = thereₛ (∈ᶠˢ-map fsuc _ (n-has _ x))

--   Finite-Σ : {P : A → Type ℓ} ⦃ _ : Finite A ⦄ ⦃ _ : ∀ {x} → Finite (P x) ⦄ → Finite (Σ A P)
--   Finite-Σ ⦃ a ⦄ ⦃ f ⦄ .univ = sigma (a .univ) (λ a → f {a} .univ)
--   Finite-Σ ⦃ a ⦄ ⦃ f ⦄ .has (x , y) = sigma-∈ᶠˢ (λ a → f {a} .univ) (a .univ) (a .has x) (f {x} .has y)

--   Finite-Lift : ⦃ _ : Finite A ⦄ → Finite (Lift ℓ A)
--   Finite-Lift ⦃ a ⦄ .univ = map lift (a .univ)
--   Finite-Lift ⦃ a ⦄ .has (lift x) = ∈ᶠˢ-map lift _ (a .has x)

-- -- xs : Finite (Fin 3 × Fin 3 × Fin 3)
-- -- xs = auto

-- -- _ = {! (λ x → (fst x , fst (snd x))) <$> test xs !}

-- --   Discrete-⊤ : Discrete ⊤
-- --   Discrete-⊤ = yes refl

-- open import Data.List.Membership
-- open import Data.List.Base hiding (count)
-- open import Data.Bool
-- import Data.List.Quantifiers as List

-- finset→list : ⦃ _ : Discrete A ⦄ (xs : Finset A) → ∃[ l ∈ List A ] ((∀ (x : A) → x ∈ xs → is-contr (x ∈ l)) × ((x : A) → x ∈ l → x ∈ xs))
-- finset→list {A = A} = go where
--   instance
--     _ : H-Level A 2
--     _ = basic-instance 2 (Discrete→is-set auto)

--   enum : (xs : Finset A) → Type _
--   enum xs = ∃[ l ∈ List A ] ((∀ (x : A) → x ∈ xs → is-contr (x ∈ l)) × ((x : A) → x ∈ l → x ∈ xs))

--   enump : (xs : Finset A) → is-prop (enum xs)
--   enump xs = squash

--   enum-cons : (x : A) (xs : Finset A) → enum xs → enum (x ∷ xs)
--   enum-cons x xs (inc (l , find , dinf)) with holds? (x ∈ xs)
--   ... | yes in-xs = inc (l , find' , dinf') where
--     find' : (y : A) → y ∈ᶠˢ (x ∷ xs) → is-contr (y ∈ₗ l)
--     find' x = ∈ᶠˢ-split (λ p → find x (substᵢ (_∈ xs) (symᵢ p) in-xs)) (find _)

--     dinf' : (y : A) → y ∈ₗ l → y ∈ᶠˢ (x ∷ xs)
--     dinf' y w = case (x ≡ᵢ? y) of λ { (yes reflᵢ) → hereₛ ; (no ¬x=y) → thereₛ (dinf y w) }
--   ... | no ¬in-xs = inc (x ∷ l , find' , dinf') where
--     find' : (y : A) → y ∈ᶠˢ (x ∷ xs) → is-contr (y ∈ₗ (x ∷ l))
--     find' y = ∈ᶠˢ-split
--       (λ p → contr (here p) (λ { (here q) → ap here prop! ; (there q) → absurd (¬in-xs (dinf _ (substᵢ (_∈ l) p q))) }))
--       (λ w → let c = find _ w in contr (there (c .centre)) (λ { (here p) → absurd (¬in-xs (dinf x (substᵢ (_∈ l) p (c .centre)))) ; (there q) → ap there (c .paths q)}))

--     dinf' : (y : A) → y ∈ₗ (x ∷ l) → y ∈ᶠˢ (x ∷ xs)
--     dinf' y (here p) = hereₛ' p
--     dinf' y (there w) = thereₛ (dinf y w)

--   enum-cons x xs (squash a b i) = squash (enum-cons x xs a) (enum-cons x xs b) i

--   go : ∀ xs → enum xs
--   go' : (x : A) (xs : Finset A) → enum (x ∷ xs)
--   go' x xs =  enum-cons x xs (go xs)

--   go [] = inc ([] , (λ x m → absurd (¬mem-[] m)) , λ a ())
--   go (x ∷ xs) = enum-cons x xs (go xs)
--   go (∷-dup x xs i) = is-prop→pathp (λ i → enump (∷-dup x xs i)) (enum-cons x (x ∷ xs) (go' x xs)) (go' x xs) i
--   go (∷-swap x y xs i) = is-prop→pathp (λ i → enump (∷-swap x y xs i)) (enum-cons x _ (go' y xs)) (enum-cons y _ (go' x xs)) i
--   go (squash x y p q i j) = is-prop→squarep (λ i j → enump (squash x y p q i j)) (λ i → go x) (λ i → go (p i)) (λ i → go (q i)) (λ i → go y) i j

-- -- pi : {P : A → Type ℓ} (xs : Finset A) → (∀ x → Finset (P x)) → Finset ((x : A) → x ∈ xs → P x)
-- -- pi [] p = pure λ _ m → absurd (¬mem-[] m)
-- -- pi (x ∷ xs) p = p x >>= λ a → map (λ f x m → {! ∈ᶠˢ-split m ? ? !}) (pi xs p)
-- -- pi (∷-dup x xs i) p = {!   !}
-- -- pi (∷-swap x y xs i) p = {!   !}
-- -- pi (squash xs xs₁ x y i i₁) p = {!   !}

-- -- Prod : List A → (P : A → Type ℓ) → Type (level-of A ⊔ ℓ)
-- -- Prod []       P = Lift _ ⊤
-- -- Prod (x ∷ xs) P = P x × Prod xs P

-- -- module _ (P : A → Type ℓ) where
-- --   prod→part : (xs : List A) → Prod xs P → (x : A) → x ∈ xs → P x
-- --   prod→part .(_ ∷ _) (pa , pr) a (here q)  = subst P (sym q) pa
-- --   prod→part .(_ ∷ _) (pa , pr) a (there w) = prod→part _ pr a w

-- --   part→prod : (xs : List A) → ((x : A) → x ∈ xs → P x) → Prod xs P
-- --   part→prod [] f = lift tt
-- --   part→prod (x ∷ xs) f = f x (here refl) , part→prod xs (λ a w → f a (there w))

-- --   prod→part→prod : (xs : List A) (p : Prod xs P) → part→prod xs (prod→part xs p) ≡ p
-- --   prod→part→prod [] p = refl
-- --   prod→part→prod (x ∷ xs) (px , pr) = transport-refl px ,ₚ prod→part→prod xs pr

-- --   part→prod→part : (xs : List A) (f : (x : A) → x ∈ xs → P x) (x : A) (w : x ∈ xs) → prod→part xs (part→prod xs f) x w ≡ f x w
-- --   part→prod→part (x ∷ xs) f x' (here p) = J' (λ a b q → (f : (y : A) → y ∈ₗ (b ∷ xs) → P y) → subst P (sym q) (f b (here refl)) ≡ f a (here q)) (λ x f → transport-refl _) p f
-- --   part→prod→part (_ ∷ xs) f x (there w) = part→prod→part xs (λ y w → f y (there w)) x w

-- --   finite-prod : (xs : List A) → (∀ x → Finite (P x)) → Finite (Prod xs P)
-- --   finite-prod [] f = auto
-- --   finite-prod (x ∷ xs) f = Finite-Σ ⦃ f x ⦄ ⦃ λ {_} → finite-prod xs f ⦄

-- -- instance
-- --   Finite-Π : {P : A → Type ℓ} ⦃ _ : Finite A ⦄ ⦃ _ : Discrete A ⦄ ⦃ _ : ∀ {x} → Finite (P x) ⦄ → Finite ((x : A) → P x)
-- --   Finite-Π {A = A} {P = P} ⦃ fa ⦄ ⦃ da ⦄ = done where
-- --     instance
-- --       _ : H-Level A 2
-- --       _ = basic-instance 2 (Discrete→is-set auto)

-- --     done : Finite ((x : A) → P x)
-- --     done = ∥-∥-out! do
-- --       (l , find , _) ← finset→list (fa .univ)

-- --       let
-- --         fin-prod : Finite (Pi l P)
-- --         fin-prod = finite-prod P l (λ _ → auto)

-- --         to : Prod l P → (x : A) → P x
-- --         to p a = prod→part P l p a (find a (fa .has a) .centre)

-- --         univ' : Finset ((x : A) → P x)
-- --         univ' = map to (fin-prod .univ)

-- --         find' : (f : ∀ x → P x) → f ∈ univ'
-- --         find' f =
-- --           let
-- --             ix  = fin-prod .has (part→prod P l (λ a _ → f a))
-- --             ix' = ∈ᶠˢ-map to (fin-prod .univ) ix
-- --           in subst (_∈ᶠˢ univ') (funext λ a → part→prod→part P l (λ a _ → f a) a (find a (fa .has a) .centre)) ix'

-- --       pure (record { univ = univ' ; has = find' })

-- -- module _ {P : A → Type ℓ} ⦃ fa : Finite A ⦄ ⦃ _ : ∀ {x} → Discrete (P x) ⦄ {f g : ∀ x → P x} where
-- --   private
-- --     instance
-- --       _ : ∀ {x} → H-Level (P x) 2
-- --       _ = basic-instance 2 (Discrete→is-set auto)

-- --     to : ⌞ finset-all (λ x → f x ≡ g x) (fa .univ) ⌟ → f ≡ g
-- --     to xs = funext λ a → finset-all-ix (λ x → f x ≡ g x) (fa .univ) xs (fa .has a)

-- --     from : f ≡ g → ⌞ finset-all (λ x → f x ≡ g x) (fa .univ) ⌟
-- --     from h = to-finset-all (λ x → f x ≡ g x) (fa .univ) λ {x} _ → happly h x

-- --     dec : Dec ⌞ finset-all (λ x → f x ≡ g x) (fa .univ) ⌟
-- --     dec = finset-all-dec (λ x → f x ≡ g x) (fa .univ)

-- --   instance
-- --     Discrete-Π : Dec (f ≡ g)
-- --     Discrete-Π with dec
-- --     ... | yes p = yes (to p)
-- --     ... | no ¬p = no λ h → ¬p (from h)
-- ```
