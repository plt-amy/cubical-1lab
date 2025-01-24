open import 1Lab.Prelude

open import Data.List.Membership
open import Data.List.Sigma
open import Data.List.Base
open import Data.Fin.Product
open import Data.Fin.Base

module Data.List.Pi where

private variable
  ℓ ℓ' : Level
  A B C : Type ℓ
  P Q : A → Type ℓ

Pi : (xs : List A) → (A → Type ℓ') → Type _
Pi {ℓ' = ℓ'} xs P = Πᶠ {length xs} {λ _ → ℓ'} (λ i → P (xs ! i))

pi : (xs : List A) (ys : ∀ a → List (P a)) → List (Pi xs P)
pi [] ys = tt ∷ []
pi (x ∷ xs) ys = sigma (ys x) (λ _ → pi xs ys)

Pi' : (xs : List A) → (A → Type ℓ') → Type _
Pi' xs P = ∀ a → a ∈ₗ xs → P a

module _ {A : Type ℓ} (P : A → Type ℓ') where
  to-pi' : ∀ {xs} → Pi xs P → Pi' xs P
  to-pi' (x , xs) a (here p) = subst P (sym p) x
  to-pi' (x , xs) a (there h) = to-pi' xs a h

  from-pi' : ∀ {xs} → Pi' xs P → Pi xs P
  from-pi' {[]} x = tt
  from-pi' {x₁ ∷ xs} x = x _ (here refl) , from-pi' {xs} (λ _ h → x _ (there h))

  to-from-pi' : ∀ {xs} (p : Pi xs P) → from-pi' (to-pi' p) ≡ p
  to-from-pi' {[]} p = refl
  to-from-pi' {x ∷ xs} p = transport-refl _ ,ₚ to-from-pi' {xs} (p .snd)

  from-to-pi' : ∀ {xs} (p : Pi' xs P) → to-pi' (from-pi' p) ≡ p
  from-to-pi' p i a (here q) = transp (λ j → P (q (~ i ∧ ~ j))) i (p (q (~ i)) (here λ j → q (~ i ∨ j)))
  from-to-pi' p i a (there x) = from-to-pi' (λ a h → p a (there h)) i a x

  pi-member'
    : {A : Type ℓ} {P : A → Type ℓ'} (xs : List A) (ys : ∀ a → List (P a)) {e : Pi xs P}
    → (∀ h → indexₚ e h ∈ₗ ys (xs ! h)) → e ∈ₗ pi xs ys
  pi-member' [] ys {tt} p = here refl
  pi-member' (x ∷ xs) ys {p , ps} q = sigma-member {xs = ys x} {ys = λ _ → pi xs ys} (q fzero) (pi-member' xs ys (q ∘ fsuc))

  member-pi'
    : {A : Type ℓ} {P : A → Type ℓ'} (xs : List A) (ys : ∀ a → List (P a)) {e : Pi xs P}
    → (α : e ∈ₗ pi xs ys) → fibre (pi-member' xs ys) α
  member-pi' [] ys (here p) = (λ h → absurd (Fin-absurd h)) , refl
  member-pi' (x ∷ xs) ys elt =
    let
      α = member-sigmaₗ (ys x) (λ _ → pi xs ys) elt
      β = member-sigmaᵣ (ys x) (λ _ → pi xs ys) elt

      (f , coh) = member-pi' xs ys β

      g = Fin-cases α λ h → member-pi' xs ys β .fst h
    in g , ap (sigma-member α) coh ∙ member-sigma-view (ys x) (λ _ → pi xs ys) elt .snd
