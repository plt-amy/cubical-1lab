<!--
```agda
open import 1Lab.Classical
open import 1Lab.Prelude

open import Data.Fin.Indexed
open import Data.List.Membership
open import Data.List.Base
open import Data.Power
open import Data.Bool
open import Data.Dec
open import Data.Fin
open import Data.Nat
open import Data.Sum

open import Logic.Propositional.Classical

open import Meta.Brackets
```
-->

```agda
module Logic.Propositional.Classical.Compactness where
```

# Compactness of classical propositional logic is a constructive taboo

The **compactness** property, for a logical system, says that we can
determine the [[satisfiability|satisfiable]] of a set $\Phi$ of formulae
--- no matter how big $\Phi$ is! --- by checking the truth of every one
of its [[*finite*]] subsets, $\Psi \sube \Phi$.

<!--
```agda
private variable
  ℓ : Level
  Γ Δ Θ : Nat
  ψ θ ζ : Ctx Γ
  P Q R : Proposition Γ
```
-->

```agda
subset-is-sat : ∀ {Γ} → (Proposition Γ → Ω) → Type _
subset-is-sat {Γ} ϕs =
  ∃[ ρ ∈ (Fin Γ → Bool) ]
  ((ϕ : Proposition Γ) → ϕ ∈ ϕs → ⟦ ϕ ⟧ ρ ≡ true)

has-compactness : Type
has-compactness =
  ∀ {Γ}
  → (ϕs : Proposition Γ → Ω)
  → (∀ (ϕs' : Proposition Γ → Ω) → ϕs' ⊆ ϕs → Finite (∫ₚ ϕs') → subset-is-sat ϕs')
  → subset-is-sat ϕs
```

Keep in mind that compactness is *not* immediate, even classically,
since the set $\Phi$ of formulae we are testing might be arbitrarily
sized --- it could be countably infinite, equinumerous with $\bR$, or
worse --- and so it does not necessarily satisfy any boundedness
principles that would allow us to "patch together" pieces of truth ---
or even carve out those pieces in the first place.

In fact, while compactness may sound like a natural statement, we will
show that this property is *too strong* to ask of classical
propositional logic, in our neutral meta-theory: if compactness holds,
then so does [[weak excluded middle]]. Here, we formalise an argument
due to [@Saving:2023].

```agda
compactness→wlem : has-compactness → WLEM
compactness→wlem compact P = ¬P∨¬¬P where
```

We fix an arbitrary proposition $P : \Omega$ _of Agda_, our meta-logic,
and an atom $x$ of our object logic. We will apply compactness to the
set

$$\{ x \mid P \} \cup \{ \lnot x \mid \lnot P \}\text{.}$$

of formulae.

```agda
  x : Proposition 1
  x = atom 0

  [x∣P] : Proposition 1 → Ω
  [x∣P] ϕ = elΩ ((x ≡ ϕ) × ∣ P ∣)

  [¬x∣¬P] : Proposition 1 → Ω
  [¬x∣¬P] ϕ = elΩ ((“¬” x ≡ ϕ) × (¬ ∣ P ∣))
```

Despite the previous examples of bad subset being extremely large, this
is actually quite tame! It only has three possible _finite_ subsets,
namely the empty set, $\{ x \}$, and $\{ \lnot x \}$, all of which are
easily seen to be satisfiable. However, showing this to a proof
assistant requires quite a bit of work in combinatorics, so the actual
formalisation is a slightly nightmarish case bash.

```agda
  finitely-consistent
    : (ϕs' : Proposition 1 → Ω)
    → ϕs' ⊆ ([x∣P] ∪ [¬x∣¬P])
    → Finite (∫ₚ ϕs')
    → subset-is-sat ϕs'
```

<details>
<summary>
We will set the `<details>`{.html} aside for the curious reader.
</summary>

```agda
  finitely-consistent ϕs' sub p = do
    li ← p
    let
      it = case li .Listing.univ return (λ v → (∀ x → is-contr (x ∈ₗ v)) → subset-is-sat ϕs') of λ where
        []              h → pure ((λ _ → true) , λ ϕ m → case h (ϕ , m) of λ ())

        ((p , hp) ∷ []) h → case sub p hp of λ where
          (inl a) → case a of λ where
            tr _ → pure ((λ _ → true) , λ φ hφ → case h (φ , hφ) .centre of λ where
              (here reflᵢ) → subst (λ e → ⟦ e ⟧ (λ _ → true) ≡ true) tr refl)
          (inr a) → case a of λ where
            tr _ → pure ((λ _ → false) , λ φ hφ → case h (φ , hφ) .centre of λ where
              (here reflᵢ) → subst (λ e → ⟦ e ⟧ (λ _ → false) ≡ true) tr refl)

        ((p , hp) ∷ (q , hq) ∷ ps) → case sub p hp , sub q hq of λ where
          (inl a) (inl b) h → do
            (a , _) ← □-tr a
            (b , _) ← □-tr b
            pure
              let
                q = Id≃path.from (Σ-prop-path! (sym a ∙ b))
                it = is-contr→is-prop (h (p , hp)) (here reflᵢ) (there (here q))
              in absurd (here≠there it)
          (inl a) (inr b) h → □-tr do
            (_ , a)  ← a
            (_ , ¬a) ← b
            absurd (¬a a)
          (inr a) (inl b) h → □-tr do
            (_ , ¬a) ← a
            (_ , a)  ← b
            absurd (¬a a)
          (inr a) (inr b) h → do
            (a , _) ← □-tr a
            (b , _) ← □-tr b
            pure
              let
                q = Id≃path.from (Σ-prop-path! (sym a ∙ b))
                it = is-contr→is-prop (h (p , hp)) (here reflᵢ) (there (here q))
              in absurd (here≠there it)
    it (λ a → li .Listing.has-member a)
```

</details>

By compactness, we must have a satisfying assignment for the set of
formulae

$$\{ x \mid P \} \cup \{ \lnot x \mid \lnot P \}\text{,}$$

which actually consists only of the boolean value assigned to the atom
$x$: something we can do case analysis on. If our compactness oracle
thinks that $x$ must be true, then it *can not have been the case that
$\lnot P$ holds*, since then we would have $x = \operatorname{false}$.
In other words, we have decided $\lnot \lnot P$.

```agda
  x-true→¬¬P
    : (ρ : Fin 1 → Bool)
    → (∀ ϕ → ϕ ∈ [x∣P] ∪ [¬x∣¬P] → sem-prop ϕ ρ ≡ true)
    → ⟦ x ⟧ ρ ≡ true
    → ¬ ¬ ∣ P ∣
  x-true→¬¬P ρ sat x-true ¬p =
    not-no-fixed $
    ⟦ x ⟧ ρ       ≡⟨ x-true ⟩
    true          ≡˘⟨ sat (“¬” x) (inc (inr (inc (refl , ¬p)))) ⟩
    not (⟦ x ⟧ ρ) ∎
```

Conversely, if $x$ was assigned false, then $\lnot P$ must not have
failed: but the triple negative $\lnot \lnot \lnot P$ can be reduced to
a single negative, even constructively.

```agda
  x-false→¬P
    : (ρ : Fin 1 → Bool)
    → (∀ ϕ → ϕ ∈ [x∣P] ∪ [¬x∣¬P] → sem-prop ϕ ρ ≡ true)
    → ⟦ x ⟧ ρ ≡ false
    → ¬ ∣ P ∣
  x-false→¬P ρ sat x-false p =
    not-no-fixed $
    ⟦ x ⟧ ρ       ≡⟨ sat x (inc (inl (inc (refl , p)))) ⟩
    true          ≡˘⟨ ap not x-false ⟩
    not (⟦ x ⟧ ρ) ∎
```

If we put this all together, then we can decide $\neg P$!

```agda
  ¬P∨¬¬P : Dec (¬ ∣ P ∣)
  ¬P∨¬¬P = ∥-∥-out! do
    (ρ , ρ-sat) ← compact ([x∣P] ∪ [¬x∣¬P]) finitely-consistent
    pure $
      Bool-elim (λ b → ρ 0 ≡ b → Dec (¬ ∣ P ∣))
        (λ x-true → no (x-true→¬¬P ρ ρ-sat x-true))
        (λ x-false → yes (x-false→¬P ρ ρ-sat x-false))
        (ρ 0) refl
```
