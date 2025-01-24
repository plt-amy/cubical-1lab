open import 1Lab.Prelude

open import Data.List.Membership
open import Data.List.Base
open import Data.Fin.Base
open import Data.Nat.Base as Nat
open import Data.Sum.Base
open import Data.Irr

module Data.List.Sigma where

module _ {ℓ ℓ'} {A : Type ℓ} {B : A → Type ℓ'} where
  private
    fibre' : {x y : A} (xs : List (B x)) (p : y ≡ x) → B y → Type _
    fibre' xs p b = Σ[ ix ∈ Fin (length xs) ] (PathP (λ i → B (p (~ i))) (xs ! ix) b)

    pair-mem : (x : A) (xs : List (B x)) (a : A) (b : B a) → Type _
    pair-mem x xs a b = Σ[ p ∈ a ≡ x ] (fibre' xs p b)

    pair-member
      : (x : A) (xs : List (B x)) (a : A) (b : B a)
      → (a , b) ∈ₗ map (x ,_) xs
      → pair-mem x xs a b
    pair-member x (y ∷ ys) a b (here p) = ap fst p , fzero , ap snd (sym p)
    pair-member x (y ∷ ys) a b (there p) with pair-member x ys a b p
    ... | (p , ix , q) = p , fsuc ix , q

    member-pair
      : (x : A) (xs : List (B x)) (a : A) (b : B a)
      → pair-mem x xs a b
      → (a , b) ∈ₗ map (x ,_) xs
    member-pair x (y ∷ ys) a b (p , i , q) with fin-view i
    ... | zero = here λ i → p i , q (~ i)
    ... | suc i = there (member-pair x ys a b (p , i , q))

    member-pair-inv
      : (x : A) (xs : List (B x)) (a : A) (b : B a) (it : (a , b) ∈ₗ map (x ,_) xs)
      → member-pair x xs a b (pair-member x xs a b it) ≡ it
    member-pair-inv x (y ∷ ys) a b (here p)   = refl
    member-pair-inv x (y ∷ ys) a b (there it) = ap there (member-pair-inv x ys a b it)

    opaque
      rem₀
        : ∀ {x a} (ys : ∀ a → List (B a)) (b : B _) (p : a ≡ x) ix .{q} .{q'}
        → PathP (λ i → B (p (~ i))) (ys x ! fin ix ⦃ forget q ⦄) b ≃ (ys a ! fin ix ⦃ forget q' ⦄ ≡ b)
      rem₀ {x = x} {a} ys b p ix {q} {q'} = J
        (λ x p → ∀ .q .q' → PathP (λ i → B (p (~ i))) (ys x ! fin ix ⦃ forget q ⦄) b ≃ (ys a ! fin ix ⦃ forget q' ⦄ ≡ b))
        (λ q q' → id , id-equiv)
        p q q'

    rem₁ : ∀ {x a} (ys : ∀ a → List (B a)) (b : B _) (p : a ≡ x) → fibre' (ys x) p b → fibre (ys a !_) b
    rem₁ {x = x} {a} ys b p (fin ix ⦃ forget q ⦄ , r) = fin ix ⦃ q' ⦄ , Equiv.to (rem₀ ys b p ix) r where
      q' = forget (transport (λ i → suc ix Nat.≤ length (ys (p (~ i)))) q)

    rem₂ : ∀ {x a} (ys : ∀ a → List (B a)) (b : B _) (p : a ≡ x) → fibre (ys a !_) b → fibre' (ys x) p b
    rem₂ {x = x} {a} ys b p (fin ix ⦃ forget q ⦄ , r) = fin ix ⦃ q' ⦄ , Equiv.from (rem₀ ys b p ix) r where
      q' = forget (transport (λ i → suc ix Nat.≤ length (ys (p i))) q)

  sigma-member : ∀ {a b xs ys} → a ∈ₗ xs → b ∈ₗ ys a → (a , b) ∈ₗ sigma xs ys
  sigma-member {a = a} {b = b} {xs = x ∷ xs} {ys = ys} (here {x' = x'} p) q =
    ++-memberₗ $ member-pair x (ys x) a b (p , rem₂ ys b p (member→lookup q))
  sigma-member (there p) q = ++-memberᵣ (sigma-member p q)

  private
    split : ∀ {a b} xs ys → (a , b) ∈ₗ sigma xs ys → Type _
    split {a = a} {b} xs ys top = Σ[ p ∈ a ∈ₗ xs ] Σ[ q ∈ b ∈ₗ ys a ] (sigma-member p q ≡ top)

    here-sigma
      : ∀ (a : A) b xs ys {x : A} (q : (a , b) ∈ₗ map (x ,_) (ys x))
      → split (x ∷ xs) ys (++-memberₗ q)
    here-sigma a b xs ys {x} p with inspect (pair-member x (ys x) a b p)
    ... | (p' , fin ix ⦃ forget q ⦄ , r) , prf = here p' , elt , coh where
      q' = forget (transport (λ i → suc ix Nat.≤ length (ys (p' (~ i)))) q)

      elt = lookup→member (rem₁ ys b p' (fin ix ⦃ forget q ⦄ , r))

      abstract
        coh : sigma-member {xs = x ∷ xs} (here p') elt ≡ ++-memberₗ p
        coh = ap ++-memberₗ
          (ap (member-pair x (ys x) a b)
            ( Σ-pathp refl (ap (rem₂ ys b p') (Equiv.ε member≃lookup _)
            ∙ Σ-pathp refl (Equiv.η (rem₀ ys b p' ix) _))
            ∙ sym prf)
          ∙ member-pair-inv x (ys x) a b p)

    member-sigma : ∀ a b xs ys (top : (a , b) ∈ₗ sigma xs ys) → split xs ys top
    member-sigma a b (x ∷ xs) ys top with member-++-view (map (x ,_) (ys x)) _ top
    ... | inl (q , prf) =
      let (a , b , it) = here-sigma a b xs ys q
      in a , b , it ∙ prf
    ... | inr (q , prf) =
      let (a , b , it) = member-sigma a b xs ys q
      in there a , b , ap ++-memberᵣ it ∙ prf

  member-sigmaₗ : ∀ {a b} xs ys → (a , b) ∈ₗ sigma xs ys → a ∈ₗ xs
  member-sigmaₗ _ _ p = member-sigma _ _ _ _ p .fst

  member-sigmaᵣ : ∀ {a b} xs ys → (a , b) ∈ₗ sigma xs ys → b ∈ₗ ys a
  member-sigmaᵣ xs ys p = member-sigma _ _ xs ys p .snd .fst

  sigma-member' : ∀ {a b xs ys} → a ∈ₗ xs × b ∈ₗ ys a → (a , b) ∈ₗ sigma xs ys
  sigma-member' (p , q) = sigma-member p q

  member-sigma-view : ∀ {a b} xs ys (p : (a , b) ∈ₗ sigma xs ys) → fibre sigma-member' p
  member-sigma-view xs ys p = record
    { fst = member-sigmaₗ xs ys p , member-sigmaᵣ xs ys p
    ; snd = member-sigma _ _ xs ys p .snd .snd
    }
