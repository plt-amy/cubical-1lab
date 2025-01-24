<!--
```agda
open import 1Lab.Prelude

open import Data.Dec.Base
open import Data.Power
open import Data.Fin
open import Data.Sum
```
-->

```agda
module Data.Fin.Indexed where
```

<!--
```agda
private variable
  ℓ : Level
  T S : Type ℓ
  P Q : ℙ S
```
-->

# Finitely-indexed sets {defines="kuratowski-finite-subset finitely-indexed-set"}

Constructively, [[finiteness|finite]] is a fairly strong condition to
impose on a type. To say that a type $X$ has a cardinality is to say it
is equipped with $n : \bN$ and [[merely]] admits a bijection $X \simeq
[n]$. This is weaker than asking it to be _equipped with_ such a
bijection, since it does not impose an [[order|partial order]] on $X$,
but it can still be used to transport any [[property]] of $[n]$ to $X$.
For example, any finite type is [[discrete]].

The strength of finiteness means that it lacks several closure
properties that are familiar from classical mathematics, the most
shocking being that _subsets of a finite set need not be finite_. This
is because every $\phi : \Omega$ determines a subset of $[1]$, whose
cardinality --- either zero or one --- serves as a decision for $\phi$.
Since it's not possible to decide arbitrary propositions, we can not
assign a cardinality to arbitrary subsets. They are not closed under
[[quotients]] either: we may quotient the set ${0,1}$ by the relation

$$
(0 \sim 1) := \phi
$$,

and this quotient has cardinality $2$ iff. $\neg \phi$.

To recover these closure properties, we may either weaken the
construction we wish to close under --- for example, limiting ourselves
to the _decidable_ subsets or quotients --- or we may weaken the
definition of finite, asking only that the type have a cardinality
bounded above by some natural number.

If we weaken the equivalence $[n] \simeq X$ to a [[surjection|surjection
between sets]] $[n] \epi X$, we recover closure under quotients. These
are the **finitely-indexed types**, also known as $K$-finite types,
after the mathematician Kuratowski.

To simplify the formalisation, we will stratify the definition of
finitely-indexed by introducing a name for the intermediate data of the
cardinality $n : \bN$ and the surjection $[n] \to X$: a **finite cover**
of $X$.

```agda
record Finite-cover {ℓ} (A : Type ℓ) : Type ℓ where
  constructor covering
  field
    {card}   : Nat
    cover    : Fin card → A
    is-cover : is-surjective cover

is-finitely-indexed : ∀ {ℓ} → Type ℓ → Type
is-finitely-indexed T = □ (Finite-cover T)

is-K-finite : (T → Ω) → Type
is-K-finite P = is-finitely-indexed (∫ₚ P)
```

<!--
```agda
{-# INLINE covering #-}
```
-->

We will use the terminology **finitely-indexed** to refer to types.
However, it's also convenient to have a name for this notion applied to
_subtypes_. We will call a predicate $P : X \to \Omega$ with
finitely-indexed total space **$K$-finite**.

::: terminology
Types are **finitely indexed** when they are [[merely]] finitely
covered, and subtypes are K-finite when their total space is
finitely-indexed.
:::

## Closure properties

As mentioned above, the key property that distinguishes the
finitely-indexed types from the finite sets is that the former are
closed under arbitrary quotients.

```agda
surjection→finite-cover
  : S ↠ T
  → Finite-cover S
  → Finite-cover T
surjection→finite-cover (f , surj) (covering g surj') =
  covering (f ∘ g) (∘-is-surjective surj surj')

surjection→is-finitely-indexed
  : S ↠ T → is-finitely-indexed S → is-finitely-indexed T
surjection→is-finitely-indexed f = map (surjection→finite-cover f)
```

Moreover, since any $f : [n] \equiv T$ can be weakened to a $f : [n]
\epi T$, any [[finite]] type is finitely-indexed, too.

```agda
Finite→is-finitely-indexed : ⦃ _ : Finite T ⦄ → is-finitely-indexed T
Finite→is-finitely-indexed ⦃ eqv ⦄ = do
  li ← tr-□ eqv
  let eqv = Equiv.inverse (Listing.listing→fin-equiv li)
  pure (covering _ (is-equiv→is-surjective (Equiv.inverse eqv .snd)))
```

## Finitely-indexed subsets

We now turn to the question of finitely-indexed _subsets_. A subset $P :
T \to \Omega$ is finitely-indexed if its total space $\Sigma_{x : T} x
\in P$ is. To disambiguate and keep names short, we refer to types as
finitely-indexed, but to subsets as **K-finite**.

The $K$-finite subsets of $T$ are a sub-[[join semilattice]] of the
[[power-set]] lattice of $T$. That is, the empty subset is $K$-finite,
and any union is $K$-finite as well. Since the empty subset is _finite_,
it's also $K$-finite, of course.

```agda
opaque
  minimal-is-K-finite : is-K-finite (minimal {X = T})
  minimal-is-K-finite = inc (covering {card = 0} (λ ()) λ ())
```

But the case of unions is slightly different. Unless $A$ is assumed to
have decidable equality, we can not compute the cardinality of the union
$P \cup Q$, since we can't decide whether or not $P$ and $Q$ have some
intersection --- and we have placed no such assumption on $A$. That's
why $K$-finite subsets express an _upper bound_ on the cardinality of a
(sub)set.

We're justified in doing this by the following observation: If we had a
magical cardinality function $n(-) : (T \to \Omega) \to \bN$, then we
would have the familiar formula

$$
n(A \cup B) = (n(A) + n(B)) - n(A \cap B)
$$.

Even though we don't, if $a$ and $b$ are upper bounds for the size of
$A$ and $B$ (i.e. $n(A) \le a$ and $n(B) \le B$), then we certainly have
$n(A \cup B) \le a + b$. The fact that our covering might over-count is
not an issue here.

```agda
  union-is-K-finite : is-K-finite P → is-K-finite Q → is-K-finite (P ∪ Q)
  union-is-K-finite {P = P} {Q = Q} p-fin q-fin = do
    covering {Pn} f f-surj ← p-fin
    covering {Qn} g g-surj ← q-fin

    let
      cover' : (Fin Pn ⊎ Fin Qn) → ∫ₚ (P ∪ Q)
      cover' = λ where
        (inl x) → f x .fst , inc (inl (f x .snd))
        (inr x) → g x .fst , inc (inr (g x .snd))

      surj' : ∀ x → ∥ fibre cover' x ∥
      surj' xf = (xf .snd) >>= λ where
        (inl p) → do
          (ix , p) ← f-surj (xf .fst , p)
          pure (inl ix , Σ-prop-path! (ap fst p))
        (inr q) → do
          (ix , p) ← g-surj (xf .fst , q)
          pure (inr ix , Σ-prop-path! (ap fst p))

    surjection→is-finitely-indexed (cover' , surj') Finite→is-finitely-indexed
```

Moreover, we have that a singleton set is $K$-finite, as well.

```agda
  singleton-is-K-finite : is-set T → (x : T) → is-K-finite (singleton x)
  singleton-is-K-finite t-set x = inc (covering {card = 1} (λ _ → x , inc refl)
    λ (y , p) → inc (fzero , Σ-prop-path! (□-rec (t-set _ _) id p)))
```
