<!--
```agda
open import 1Lab.Path
open import 1Lab.Type

open import Data.Maybe
open import Data.Bool
open import Data.Dec
open import Data.Fin
open import Data.Nat
open import Data.Sum
open import Data.Vec.Base
```
-->

```agda
module ProgrammingLanguage.STLC.DebrujinExtrinsic where
```

# STLC, with Debrujin indexes

In the [previous episode of this series], we explored the STLC using
names (in our case, natural numbers) to identify variables. This turns
out to not be that great, unfortunatly - I invite you to go look
at some of the detail blocks, and gaze upon the not-so-great code
there. In order to avoid some of this, we're going to explore using
**Debrujin indexes**, which provide a method of avoiding these
pain points.

Debrujin indexes are, in short, a way of identifying variables without
using names. An index $n$ identifies the $n$th binder "upwards" -
for example, in `λ₀ 0`, the `0` refers to the `0`th binder, labeled
`λ₀`. In `λ₁ (λ₀ (0 1))`, `0` refers to the innermost binder, `λ₀`,
and `1` refers to the next-upwards, `λ₁`. A useful way of picturing
this is that if binders were a cons list, the Debrujin index refers
to an index into that list. Under this detail block are some more
"usual" terms, and their respective Debrujin implementations, as a
learning tool:

<details>
```agda
-- const = λ a. λ b. b
-- constd = λ λ 0

-- app = λ f. λ x. f x
-- app = λ λ 1 0
```
When you enter another binder, everything existing gets "pushed"
up one level:
```
-- weird = λ f. λ x. f (λ y. y x) x
-- weird = λ λ 1 (λ 0 1) 0
```
</details>

[previous episode of this series]: ProgrammingLanguage.STLC.Simple.lagda.html

First off, our types are the same as the previous entry:

```agda
data Ty : Type where
  UU : Ty
  _`×_ : Ty → Ty → Ty
  _`⇒_ : Ty → Ty → Ty
```

Contexts are now Vectors of a set length; this allows us to index
them non-partially with a number less than the length, which for a
number `n` we notate `Fin n`{.Agda}. We again use our reversed
constructor, `_∷c_`{.Agda}.

```agda
Con : Nat → Type
Con n = Vec Ty n 
```
<!--
```agda
infixl 10 _∷c_
pattern _∷c_ Γ x = x ∷ Γ
```
-->

```agda
index : ∀ {n} → Con n → Fin n → Ty
index (x ∷ Γ) n with fin-view n
... | zero = x
... | suc i = index Γ i
```

We now move on to defining our `Expr`{.Agda} type. The "magic" here
is that we index the term itself by the length of the context - this
way, it's impossible to reference a variable that doesn't exist.

```agda
data Expr (n : Nat): Type where
```
Variables are slightly different - they're now `Fin`{.Agda}s.
```agda
  ` : Fin n → Expr n
```
As lambda abstractions introduce a new variable, they take an expression
extended with another variable, and return one without that variable.
```agda
  `λ : Expr (suc n) → Expr n
```
Everything else proceeds as expected.
```agda
  `$ : Expr n → Expr n → Expr n
  `⟨_,_⟩ :  Expr n → Expr n → Expr n
  `π₁ : Expr n → Expr n
  `π₂ : Expr n → Expr n
  `U : Expr n
```

We call any expression with no "free variables" (variables not introduced
in the term itself, by lambda abstractions) a "closed term". In our
case, this is equivalent to being an expression of type `Expr 0`{.Agda},
as this enforces that without lambda abstraction, we cannot introduce
any variables - the is no element of `Fin 0`{.Agda} (as there is no
number smaller than `0`).

<!--
```agda
module Example-1 where
```
-->
The first term is closed, as it refers to no outside variables, and
equivalently has type `Expr 0`{.Agda}. The second is not, as it
refers to an outside term (and therefore does not have type `Expr 0`{.Agda}
```agda
  is-closed-app : Expr 0
  is-closed-app = `λ (`λ (`$ (` (fin 1)) (` (fin 0))))
  
  isn't-closed : Expr 1
  isn't-closed = ` 0

  -- isn't-closed-wrong : Expr 0
  -- isn't-closed-wrong = ` 0
  -- Errors
```

We again have a typing relation - this time, before
we present it, we'll elaborate a bit more on the name of this file,
and what it means to be "extrinsic". Extrinsic typing states that
expressions do not inherently have a type, and types are assigned via
some relation *extrinsic* to the expressions themselves - here,
the typing relation we're about to create. Intrinsic typing, which
will be explored in the next file, has expressions with inherently types,
*intrinsic* to them.

TODO: explain typing relation

<!--
```agda
infix 3 _⊢_⦂_
```
-->

```agda
data _⊢_⦂_ : ∀ {n} → Con n → Expr n → Ty → Type where
```
Variables are almost the same, but this time we can avoid the
partiality.
```agda
  `⊢ : ∀ {n} {Γ : Con n} {f : Fin n} {τ} →
       index Γ f ≡ τ →
       Γ ⊢ ` f ⦂ τ
```
Lambda abstractions are also similar, but we don't need to introduce
the name this time. Note how the body exists in an index one "higher"
than the resulting abstraction, to accomodate the extra introduced
variable.
```
  `λ⊢ : ∀ {n} {Γ : Con n} {bd : Expr (suc n)} {τ ρ} →
        Γ ∷c τ ⊢ bd ⦂ ρ →
        Γ ⊢ `λ bd ⦂ τ `⇒ ρ
```
The rest of the formers follow, basically unchanged.
```agda
  `·⊢ : ∀ {n} {Γ : Con n} {f x : Expr n} {τ ρ} →
        Γ ⊢ f ⦂ τ `⇒ ρ →
        Γ ⊢ x ⦂ τ →
        Γ ⊢ `$ f x ⦂ ρ
  `⟨,⟩⊢ : ∀ {n} {Γ : Con n} {a b : Expr n} {τ ρ} →
        Γ ⊢ a ⦂ τ →
        Γ ⊢ b ⦂ ρ →
        Γ ⊢ `⟨ a , b ⟩ ⦂ τ `× ρ
  `π₁ : ∀ {n} {Γ : Con n} {a : Expr n} {τ ρ} →
        Γ ⊢ a ⦂ τ `× ρ →
        Γ ⊢ `π₁ a ⦂ τ
  `π₂ : ∀ {n} {Γ : Con n} {a : Expr n} {τ ρ} →
        Γ ⊢ a ⦂ τ `× ρ →
        Γ ⊢ `π₂ a ⦂ ρ
  `U⊢ : ∀ {n} {Γ : Con n} →
        Γ ⊢ `U ⦂ UU        
```
