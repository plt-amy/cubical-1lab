---
description: |
  We define set-projective types, and prove some taboos.
---
<!--
```agda
open import 1Lab.Classical
open import 1Lab.Prelude

open import Data.Dec
open import Data.Fin

open import Meta.Invariant
```
-->
```agda
module Data.Set.Projective where
```

# Set-projective types

<!--
```agda
private variable
  ℓ ℓ' ℓ'' : Level
```
-->

:::{.definition #set-projective}
A type $A$ is **set-projective** if we can commute [[propositional truncation]]
past $A$-indexed families.
:::

```agda
is-set-projective : ∀ {ℓ} (A : Type ℓ) → (κ : Level) → Type _
is-set-projective A κ =
  ∀ (P : A → Type κ)
  → (∀ a → is-set (P a))
  → (∀ a → ∥ P a ∥)
  → ∥ (∀ a → P a) ∥
```

Intuitively, a type $A$ is set-projective if it validates a sort of
$A$-indexed version of the [[axiom of choice]].

If $A$ is a set, then $A$ is set-projective if and only if every
surjection $E \to A$ from a set $E$ splits.

```agda
surjections-split→set-projective
  : ∀ {ℓ ℓ'} {A : Type ℓ}
  → is-set A
  → (∀ (E : Type (ℓ ⊔ ℓ')) → is-set E
     → (f : E → A) → is-surjective f
     → ∥ (∀ a → fibre f a) ∥)
  → is-set-projective A (ℓ ⊔ ℓ')

sets-projective→surjections-split
  : ∀ {ℓ ℓ'} {A : Type ℓ}
  → is-set A
  → is-set-projective A (ℓ ⊔ ℓ')
  → ∀ {E : Type ℓ'} → is-set E
  → (f : E → A) → is-surjective f
  → ∥ (∀ a → fibre f a) ∥
```

<details>
<summary>This is essentially a specialization of the of the proof that
the axiom of choice is equivalent to every surjection splitting, so we
will not dwell on the details.
</summary>
```agda
surjections-split→set-projective {A = A} A-set surj-split P P-set ∥P∥ =
  ∥-∥-map
    (Equiv.to (Π-cod≃ (Fibre-equiv P)))
    (surj-split (Σ[ x ∈ A ] (P x)) (Σ-is-hlevel 2 A-set P-set) fst λ x →
      ∥-∥-map (Equiv.from (Fibre-equiv P x)) (∥P∥ x))

sets-projective→surjections-split A-set A-pro E-set f =
  A-pro (fibre f) (λ x → fibre-is-hlevel 2 E-set A-set f x)
```
</details>

## Closure of set-projectivity

Set-projective types are closed under Σ-types. Suppose that $A : \ty$ is a
set-projective type, and that $B : A \to \ty$ is a family of set-projective
types, and let $P : \Sigma\ A\ B \to \set$ be a family of merely inhabited sets.
Note that $(b : B(a)) \to P(a, b)$ is a $B(a)$-indexed family of merely
inhabited sets for every $a$, so its product must also be inhabited by projectivity
of $B(a)$. Moreover, $A$ is also projective, so $(a : A) \to (b : B(a)) \to P(a, b)$
is also merely inhabited, as $(b : B(a)) \to P(a, b)$ is an $A$-indexed family of
merely inhabited sets. We can then uncurry this family to finish the proof.

```agda
Σ-set-projective
  : ∀ {A : Type ℓ} {B : A → Type ℓ'}
  → is-set-projective A (ℓ' ⊔ ℓ'')
  → (∀ a → is-set-projective (B a) ℓ'')
  → is-set-projective (Σ[ a ∈ A ] B a) ℓ''
Σ-set-projective {A = A} {B = B} A-pro B-pro P P-set ∥P∥ = do
  ∥-∥-map uncurry $
    A-pro (λ a → ((b : B a) → P (a , b))) (λ a → Π-is-hlevel 2 λ b → P-set (a , b)) λ a →
    B-pro a (λ b → P (a , b)) (λ b → P-set (a , b)) λ b → ∥P∥ (a , b)
```

Moreover, set-projective types are stable under retracts. Suppose that
we have $f : A \to B, g : B \to A$ with $f \circ g = id$ with $A$ set-projective,
and let $P : B \to \set$ be a family of merely inhabited sets. We can
precompose $P$ with $f$ to obtain an $A$-indexed family of sets whose
product $\Pi (a : A) \to P(f(a))$ must be inhabited via projectivity of $A$.
Moreover, we can precompose again with $g$ to see that $\Pi (b : B) \to P(f(g(b)))$
is merely inhabited. Finally, $f(g(b)) = b$, so $\Pi (b : B) \to P(b)$ is
merely inhabited.

```agda
retract→set-projective
  : ∀ {A : Type ℓ} {B : Type ℓ'}
  → (f : A → B) (g : B → A)
  → is-left-inverse f g
  → is-set-projective A ℓ''
  → is-set-projective B ℓ''
retract→set-projective {A = A} {B = B} f g retract A-pro P P-set ∥P∥ =
  ∥-∥-map (λ k b → subst P (retract b) (k (g b)))
    (A-pro (P ∘ f) (P-set ∘ f) (∥P∥ ∘ f))
```

This gives us a nice proof that set-projectivity is stable under equivalence.

```agda
Equiv→set-projective
  : ∀ {A : Type ℓ} {B : Type ℓ'}
  → A ≃ B
  → is-set-projective A ℓ''
  → is-set-projective B ℓ''
Equiv→set-projective f A-pro =
  retract→set-projective (Equiv.to f) (Equiv.from f) (Equiv.ε f) A-pro
```

By the theorem of [[finite choice]], finite sets are projective.

```agda
Fin-set-projective : ∀ {n} → is-set-projective (Fin n) ℓ
Fin-set-projective {n = n} P P-set ∥P∥ = finite-choice n ∥P∥

finite→set-projective
  : {A : Type ℓ}
  → Finite A
  → is-set-projective A ℓ'
finite→set-projective finite =
  rec! (λ enum → Equiv→set-projective (enum e⁻¹) Fin-set-projective)
    (enumeration finite)
```

## Taboos

As it turns out, the finite types are the *only* types that are projective
constructively! The general sketch of the taboo is that it is consistent that:

1. Propositions are projective.
2. Every infinite set admits an injection from `Nat`{.Agda}. In other
  words, every infinite set is Dedekind-infinite.
3. Countable choice fails (IE: the natural numbers are not set-projective).

The existence of such a model is out of scope for this page, so we will
focus our attention on the internal portion of the argument. In particular,
we will prove that if propositions are set-projective, then the existence of an
Dedekind-infinite set-projective type implies countable choice.

First, note that if propositions are set-projective, then the [[image]] of
every function into a set-projective type is itself set-projective. This
follows directly from the definition of images, along with closure of
projectives under `Σ`{.Agda}.

```agda
module _
  (props-projective : ∀ {ℓ ℓ'} → (A : Type ℓ) → is-prop A → is-set-projective A ℓ')
  where

  props-projective→image-projective
    : ∀ {A : Type ℓ} {B : Type ℓ'}
    → (f : A → B)
    → is-set-projective B (ℓ ⊔ ℓ' ⊔ ℓ'')
    → is-set-projective (image f) ℓ''
  props-projective→image-projective f B-pro =
    Σ-set-projective B-pro λ b → props-projective _ (hlevel 1)
```

This in turn implies that set-projective types are stable under embeddings,
as the image of an [[embedding]] $f : A \mono B$ is equivalent to $A$.

```agda
  props-projective+is-embedding→set-projective
    : ∀ {A : Type ℓ} {B : Type ℓ'} {f : A → B}
    → is-embedding f
    → is-set-projective B (ℓ ⊔ ℓ' ⊔ ℓ'')
    → is-set-projective A ℓ''
  props-projective+is-embedding→set-projective {f = f} f-emb B-pro =
    Equiv→set-projective
      (is-embedding→image-equiv f-emb e⁻¹)
      (props-projective→image-projective f B-pro)
```

If we specialise this result to embeddings $Nat \mono A$, then we
obtain countable choice from the existence of a Dedekind-infinite type.

```agda
  props-projective+dedekind-infinite-projective→countable-choice
    : ∀ {A : Type ℓ} {f : Nat → A}
    → is-embedding f
    → is-set-projective A (ℓ ⊔ ℓ')
    → is-set-projective Nat ℓ'
  props-projective+dedekind-infinite-projective→countable-choice =
    props-projective+is-embedding→set-projective
```

Note that the set-projectivity of propositions is itself a taboo: in particular,
every proposition is set-projective if and only if every set has split support.
The following proof is adapted from [@Kraus-Escardó-Coquand-Altenkirch:2016].

We will start with the reverse direction. Suppose that every proposition
is set projective, and let $A$ be a set. The truncation of $A$ is a propositon,
and the constant family $\| A \| \to A$ is a set-indexed family, so projectivity
of $\| A \|$ directly gives us split support.

```agda
props-projective→split-support
  : ∀ {ℓ}
  → ((A : Type ℓ) → is-prop A → is-set-projective A ℓ)
  → ∀ (A : Type ℓ) → is-set A → ∥ (∥ A ∥ → A) ∥
props-projective→split-support props-projective A A-set =
  props-projective ∥ A ∥ (hlevel 1) (λ _ → A) (λ _ → A-set) id
```

For the forward direction, suppose that every set has split support,
let $A$ be a proposition, and $P : A \to \set$ a family of merely inhabited
sets. Note that the type $\Sigma A P$ is a set, so it must have split
support $s : \| \| \Sigma A P \| \to \Sigma A P \|$. Moreover, $P(a)$
is always merely inhabited, so we can readily show that $\| A \to \Sigma A P\|$.
Finally, $A$ is a proposition, so we can obtain a $P(a)$ from $\Sigma A P$
for any $a : A$; if we combine this with our previous observation, we
immediately get $\| (a : A) \to P(a) \|$.

```agda
split-support→props-projective
  : ∀ {ℓ}
  → (∀ (A : Type ℓ) → is-set A → ∥ (∥ A ∥ → A) ∥)
  → (A : Type ℓ) → is-prop A → is-set-projective A ℓ
split-support→props-projective split-support A A-prop P P-set ∥P∥ = do
  s ← split-support (Σ[ a ∈ A ] P a) (Σ-is-hlevel 2 (is-prop→is-set A-prop) P-set)
  pure λ a → subst P (A-prop _ _) (snd (s (∥-∥-map (λ p → (a , p)) (∥P∥ a))))
```
