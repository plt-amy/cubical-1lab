open import 1Lab.Type
open import 1Lab.Path

open import Data.Id.Base

module Data.Id.Properties where

private variable
  ℓ ℓ' ℓ'' : Level
  A B C : Type ℓ
  P Q R : A → Type ℓ
  x y z : A

symᵢ-symᵢ : ∀ {ℓ} {A : Type ℓ} {x y : A} (p : x ≡ᵢ y) → symᵢ (symᵢ p) ≡ p
symᵢ-symᵢ reflᵢ = refl

symᵢ-from : ∀ {ℓ} {A : Type ℓ} {x y : A} (p : x ≡ y) → symᵢ (Id≃path.from p) ≡ Id≃path.from (sym p)
symᵢ-from = J (λ y p → symᵢ (Id≃path.from p) ≡ Id≃path.from (sym p)) (ap symᵢ (transport-refl reflᵢ) ∙ sym (transport-refl reflᵢ))


apᵢ-from : (f : A → B) {x y : A} (p : x ≡ y) → apᵢ f (Id≃path.from p) ≡ Id≃path.from (ap f p)
apᵢ-from f = J (λ y p → apᵢ f (Id≃path.from p) ≡ Id≃path.from (ap f p)) (ap (apᵢ f) (transport-refl reflᵢ) ∙ sym (transport-refl reflᵢ))

apᵢ-symᵢ : (f : A → B) (p : x ≡ᵢ y) → apᵢ f (symᵢ p) ≡ᵢ symᵢ (apᵢ f p)
apᵢ-symᵢ f reflᵢ = reflᵢ

symPᵢ : {a b : A} {x : P a} {y : P b} (p : a ≡ᵢ b) → Id-over P p x y → Id-over P (symᵢ p) y x
symPᵢ reflᵢ reflᵢ = reflᵢ

symPᵢ⁻ : {a b : A} {x : P a} {y : P b} (p : a ≡ᵢ b) → Id-over P (symᵢ p) y x → Id-over P p x y
symPᵢ⁻ reflᵢ reflᵢ = reflᵢ
