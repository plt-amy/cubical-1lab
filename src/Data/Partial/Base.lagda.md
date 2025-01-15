<!--
```agda
open import 1Lab.Prelude

open import Data.Maybe.Base
open import Data.List.Base
open import Data.Nat.Base
open import Data.Dec
```
-->

```agda
module Data.Partial.Base where
```

<!--
```agda
private variable
  o o' ℓ : Level
  A B C : Type ℓ
```
-->

# The partiality monad {defines="partiality-monad partial-element"}

The **partiality monad** is an approach to modelling _partial
computations_ in constructive type theory --- computations which may
_fail_. The meaning of failure depends on the specific computation: for
example, it might be non-terminating, or undefined on a given input. The
partiality monad abstracts away from these details to arrive at the
general notion of **partial elements**.

A partial element $x$ of a type $A$ consists of a predicate $d : \Omega$
and a function $x : d \to A$. We refer to $d$ as the **extent of
definition** of the partial element, since its function is to record
precisely _where_ a given element is defined. The function contained
within produces an element $x(-) : A$ on that extent.

```agda
record ↯ {ℓ} (A : Type ℓ) : Type ℓ where
  no-eta-equality
  field
    def : Ω
    elt : ∣ def ∣ → A
```

::: warning
This notion of partial element is not to be confused with the [partial
elements] which feature in the syntax of cubical type theory!
:::

[partial elements]: 1Lab.Path.html#partial-elements

We note that, while each element of `↯`{.Agda} stores a
[[proposition]] --- and, therefore, a type --- the type $\zap A$ of
partial $A$-elements lives in the same universe level as $A$ itself.
This is because of our standing assumption of [[propositional
resizing]]. Were it not for this assumption, we would additionally need
to quantify over a level $\ell$ bounding the size of the extents,
generating an $\ell$-large type of $\ell$-partial elements.

<!--
```agda
open ↯ public

instance
  Underlying-Part : Underlying (↯ A)
  Underlying-Part = record { ℓ-underlying = lzero ; ⌞_⌟ = λ x → ⌞ x .def ⌟ }

abstract
  ↯-indep : (x : ↯ A) {p q : ⌞ x ⌟} → x .elt p ≡ x .elt q
  ↯-indep x = ap (x .elt) (x .def .is-tr _ _)

Part-pathp
  : {x : ↯ A} {y : ↯ B} (p : A ≡ B) (q : x .def ≡ y .def)
  → PathP (λ i → ∣ q i ∣ → p i) (x .elt) (y .elt)
  → PathP (λ i → ↯ (p i)) x y
Part-pathp {x = x} {y = y} p q r i .def = q i
Part-pathp {x = x} {y = y} p q r i .elt = r i
```
-->

Partial elements enjoy the following extensionality principle: $x = y$
whenever they are defined on equivalent extents, and they agree whenever
defined.

```agda
part-ext
  : {x y : ↯ A} (to : ⌞ x ⌟ → ⌞ y ⌟) (of : ⌞ y ⌟ → ⌞ x ⌟)
  → (∀ xd yd → x .elt xd ≡ y .elt yd)
  → x ≡ y
part-ext to from p = Part-pathp refl
  (Ω-ua (biimp to from)) (funext-dep λ _ → p _ _)
```

To close the initial definitions, if we have a partial element $x : \zap
A$ and a total $y : A$, then we write $x \downarrow y$ for the relation
_$x$ is defined and $x(-) = y$_:

```agda
_↓_ : ↯ A → A → Type _
x ↓ y = Σ[ d ∈ x ] (x .elt d ≡ y)
```

<!--
```agda
instance
  Membership-↯ : Membership A (↯ A) _
  Membership-↯ = record { _∈_ = λ x p → p ↓ x }
```
-->

<!--
```agda
abstract instance
  H-Level-↯ : ∀ {A : Type ℓ} {n} ⦃ _ : 2 ≤ n ⦄ ⦃ _ : H-Level A n ⦄ → H-Level (↯ A) n
  H-Level-↯ {n = suc (suc n)} ⦃ s≤s (s≤s p) ⦄ =
    hlevel-instance $ Iso→is-hlevel! (2 + n) eqv
    where unquoteDecl eqv = declare-record-iso eqv (quote ↯)
```
-->

## The information ordering {defines="information-ordering"}

The **information ordering** on partial elements embodies the idea that
$x : \zap A$ is a _computation_ valued in $A$: If we have two such
computations $x, y$, there is an emergent notion of one having made
_more progress_ than the other --- for example, if we were to model them
as Turing machines, one might have literally taken more steps than the
other.

We write the information ordering as $x \lsq y$: it holds whenever $x$
is *more defined than* $y$, and they agree on this common extent.

```agda
record _⊑_ {ℓ} {A : Type ℓ} (x y : ↯ A) : Type ℓ where
  no-eta-equality
  field
    implies : ⌞ x ⌟ → ⌞ y ⌟
    refines : ∀ xd → y .elt (implies xd) ≡ x .elt xd
```

<!--
```agda
open _⊑_ public

abstract instance
  H-Level-⊑ : ∀ {A : Type ℓ} {x y : ↯ A} {n} ⦃ _ : 1 ≤ n ⦄ ⦃ _ : H-Level A (suc n) ⦄ → H-Level (x ⊑ y) n
  H-Level-⊑ {n = suc n} ⦃ s≤s p ⦄ = hlevel-instance $ Iso→is-hlevel! (suc n) eqv
    where unquoteDecl eqv = declare-record-iso eqv (quote _⊑_)
```
-->

At the [[bottom element]] in the information order, we have the
computation which `never`{.Agda} succeeds.

```agda
never : ↯ A
never .def = ⊥Ω
```

## Monadic structure

We claimed above that the type of partial elements is a _monad_:
equipped with the information order, it's actually the [[free pointed
directed-continuous partial order|free pointed DCPO]] on a [[set]].
However, it's also a monad in the looser sense of _supporting
`do`{.Agda} notation_.

We can define the necessary components here, without getting out the
entire [[adjoint functor]] machinery. First, we note that function
composition lets us extend $A \to B$ to $\zap A \to \zap B$:

```agda
part-map : (A → B) → ↯ A → ↯ B
part-map f x .def = x .def
part-map f x .elt = f ∘ x .elt
```

Then, we can define the transformation $A \to \zap A$, and the
"extension" operator `part-bind`{.Agda}. This latter definition
highlights an important feature of $\Omega$: it is _closed under
dependent sums_. A type of propositions closed under dependent sums is
called a **dominance**.

```agda
always : A → ↯ A
always a .def   = ⊤Ω
always a .elt _ = a

part-bind : ↯ A → (A → ↯ B) → ↯ B
part-bind x f .def .∣_∣   = Σ[ px ∈ x ] (f ʻ x .elt px)
part-bind x f .def .is-tr = hlevel 1
part-bind x f .elt (px , pfx) = f (x .elt px) .elt pfx
```

We note that we can also lift the operation of function application to
the application of partial functions to partial arguments: we get a
result whenever they are both defined.

```agda
part-ap : ↯ (A → B) → ↯ A → ↯ B
part-ap f x .def .∣_∣      = ⌞ f ⌟ × ⌞ x ⌟
part-ap f x .def .is-tr    = hlevel 1
part-ap f x .elt (pf , px) = f .elt pf (x .elt px)
```

<!--
```agda
instance
  ↯-Map : Map (eff ↯)
  ↯-Map .Map.map = part-map

  ↯-Idiom : Idiom (eff ↯)
  ↯-Idiom .Idiom.Map-idiom = ↯-Map
  ↯-Idiom .Idiom.pure      = always
  ↯-Idiom .Idiom._<*>_     = part-ap

  ↯-Bind : Bind (eff ↯)
  ↯-Bind .Bind._>>=_      = part-bind
  ↯-Bind .Bind.Idiom-bind = ↯-Idiom
```
-->

## What about Maybe?

In functional programming circles, it's common to define a *partial
function* $A \to B$ to be a total function $A \to
\operatorname{Maybe}(B)$. The `Maybe`{.Agda} type is certainly simpler
than our `↯`{.Agda}: so why not use that?

The answer lies, as it so often does, in _constructivism_. The functions
$f : A \to \operatorname{Maybe}(B)$ are precisely those for which it is
[[decidable]] whether or not $f(x)$ is defined, for each $x : A$. But if
we do not accept the [[law of excluded middle]], then using the proper
partiality monad gains us actual generality.

We can prove that the type `Maybe A`{.Agda ident=Maybe} is equivalent to
the subtype of `↯ A`{.Agda ident=↯} with [[decidable]] extent, by
writing simple functions back and forth:

```agda
↯-Maybe : ∀ {ℓ} → Type ℓ → Type ℓ
↯-Maybe A = Σ[ p ∈ ↯ A ] Dec ⌞ p ⌟

maybe→↯-maybe : Maybe A → ↯-Maybe A
maybe→↯-maybe nothing  = never , auto
maybe→↯-maybe (just x) = always x , auto

↯-maybe→maybe : ↯-Maybe A → Maybe A
↯-maybe→maybe (x , yes x-def) = just (x .elt x-def)
↯-maybe→maybe (x , no ¬x-def) = nothing

maybe→↯→maybe : (x : Maybe A) → ↯-maybe→maybe (maybe→↯-maybe x) ≡ x
maybe→↯→maybe nothing  = refl
maybe→↯→maybe (just x) = refl

↯→maybe→↯ : (x : ↯-Maybe A) → maybe→↯-maybe (↯-maybe→maybe x) ≡ x
↯→maybe→↯ (x , yes a) = Σ-prop-path! $ part-ext (λ _ → a) (λ _ → tt)
  (λ _ _ → ↯-indep x)
↯→maybe→↯ (x , no ¬a) = Σ-prop-path! $ part-ext (λ ()) ¬a λ ()
```

However, we can not prove that this is a genuine generalisation. That is
because constructive type theory is _compatible with_ the law of
excluded middle, but if we could internally demonstrate an undecidable
proposition, it would _contradict_ the law of excluded middle.
