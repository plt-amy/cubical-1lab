<!--
```agda
open import 1Lab.Reflection.Solver
open import 1Lab.Reflection
open import 1Lab.Prelude hiding (id ; _∘_)

open import Cat.Base

open import Data.List
```
-->

```agda
module Cat.Solver where
```

<!--
```agda
private variable
  o h : Level
```
-->

# Solver for categories

This module is split pretty cleanly into two halves: the first half
implements an algorithm for reducing, in a systematic way, problems
involving associativity and identity of composition in a precategory.
The latter half, significantly more cursed, uses this infrastructure to
automatically solve equality goals of this form.

With a precategory in hand, we start by defining a language of composition.

```agda
module NbE (Cat : Precategory o h) where
  open Precategory Cat
```
<!--
```agda
  private variable
    A B C : Ob
```
-->
```agda
  data Expr : Ob → Ob → Type (o ⊔ h) where
    `id  : Expr A A
    _`∘_ : Expr B C → Expr A B → Expr A C
    _↑   : Hom A B → Expr A B

  infixr 40 _`∘_
  infix 50 _↑
```

A term of type `Expr`{.Agda} represents, in a symbolic way, a composite
of morphisms in our category $C$. What this means is that, while $f
\circ g$ is some unknowable inhabitant of `Hom`{.Agda}, $f `\circ g$
represents an inhabitant of `Hom`{.Agda} which is _known_ to be a
composition of (the trees that represent) $f$ and $g$. We can now define
"two" ways of computing the morphism that an `Expr`{.Agda} represents.
The first is a straightforward `embed`{.Agda}ding:

```agda
  embed : Expr A B → Hom A B
  embed `id      = id
  embed (f ↑)    = f
  embed (f `∘ g) = embed f ∘ embed g
```

<!--
```agda
  instance
    ⟦⟧-Expr : ⟦⟧-notation (Expr A B)
    ⟦⟧-Expr = brackets _ embed
```
-->

The second computation is a bit less obvious. If you're a programmer, it
should be familiar under the name "continuation passing style".
Categorically, it can be seen as embedding into the presheaf category of
$C$. In either case, the difference is that instead of computing a
single morphism, we compute a _transformation of hom-spaces_:

```agda
  eval : Expr B C → Hom A B → Hom A C
  eval `id f      = f
  eval (f ↑) g    = f ∘ g
  eval (f `∘ g) h = eval f (eval g h)

  nf : Expr A B → Hom A B
  nf e = eval e id
```

Working this out in a back-of-the-envelope calculation, one sees that
`eval f id` should compute the same morphism as `embed f`. Indeed,
that's the case! Since `embed`{.Agda} is the "intended semantics", and `eval`{.Agda} is an
"optimised evaluator", we call this result _soundness_. We can prove it
by induction on the expression, by first generalising over `id`{.Agda}:

```agda
  eval-sound-k : (e : Expr B C) (f : Hom A B) → eval e f ≡ ⟦ e ⟧ ∘ f
  eval-sound-k `id f = sym (idl _) -- f ≡ id ∘ f
  eval-sound-k (f `∘ g) h =
    eval f (eval g h)       ≡⟨ eval-sound-k f _ ⟩
    embed f ∘ eval g h      ≡⟨ ap (embed f ∘_) (eval-sound-k g _) ⟩
    embed f ∘ embed g ∘ h   ≡⟨ assoc _ _ _ ⟩
    (embed f ∘ embed g) ∘ h ∎
  eval-sound-k (x ↑) f = refl -- x ∘ f ≡ x ∘ f

  eval-sound : (e : Expr A B) → nf e ≡ ⟦ e ⟧
  eval-sound e = eval-sound-k e id ∙ idr _
```

We now have a general theorem for solving associativity and identity
problems! If two expressions compute the same transformation of
hom-sets, then they represent the same morphism.

```agda
  abstract
    solve : (f g : Expr A B) → nf f ≡ nf g → ⟦ f ⟧ ≡ ⟦ g ⟧
    solve f g p = sym (eval-sound f) ·· p ·· (eval-sound g)

    solve-filler : (f g : Expr A B) → (p : nf f ≡ nf g) → Square (eval-sound f) p (solve f g p) (eval-sound g)
    solve-filler f g p j i = ··-filler (sym (eval-sound f)) p (eval-sound g) j i
```

# The cursed part

```agda
module Reflection where

  pattern category-args xs =
    _ hm∷ _ hm∷ _ v∷ xs

  pattern “id” =
    def (quote Precategory.id) (category-args (_ h∷ []))

  pattern “∘” f g =
    def (quote Precategory._∘_) (category-args (_ h∷ _ h∷ _ h∷ f v∷ g v∷ []))

  mk-category-args : Term → List (Arg Term) → List (Arg Term)
  mk-category-args cat xs = unknown h∷ unknown h∷ cat v∷ xs

  “solve” : Term → Term → Term → Term
  “solve” cat lhs rhs = def (quote NbE.solve) (mk-category-args cat $ infer-hidden 2 $ lhs v∷ rhs v∷ def (quote refl) [] v∷ [])

  “nf” : Term → Term → Term
  “nf” cat e = def (quote NbE.nf) (mk-category-args cat $ infer-hidden 2 $ e v∷ [])

  build-expr : Term → Term
  build-expr “id” = con (quote NbE.`id) []
  build-expr (“∘” f g) = con (quote NbE._`∘_) (build-expr f v∷ build-expr g v∷ [] )
  build-expr f = con (quote NbE._↑) (f v∷ [])

  dont-reduce : List Name
  dont-reduce = quote Precategory.id ∷ quote Precategory._∘_ ∷ []

  cat-solver : Term → SimpleSolver
  cat-solver cat .SimpleSolver.dont-reduce = dont-reduce
  cat-solver cat .SimpleSolver.build-expr tm = pure $ build-expr tm
  cat-solver cat .SimpleSolver.invoke-solver = “solve” cat
  cat-solver cat .SimpleSolver.invoke-normaliser = “nf” cat

  repr-macro : Term → Term → Term → TC ⊤
  repr-macro cat f _ =
    mk-simple-repr (cat-solver cat) f

  simplify-macro : Term → Term → Term → TC ⊤
  simplify-macro cat f hole =
    mk-simple-normalise (cat-solver cat) f hole

macro
  repr-cat! : Term → Term → Term → TC ⊤
  repr-cat! cat f = Reflection.repr-macro cat f

  simpl-cat! : Term → Term → Term → TC ⊤
  simpl-cat! cat f = Reflection.simplify-macro cat f
```

<!--
```agda
module _ {o h} (C : Precategory o h) {x y : ⌞ C ⌟} {h1 h2 : C .Precategory.Hom x y} where
  open Reflection

  private
    cat-worker : Term → TC ⊤
    cat-worker goal = withReconstructed true $ withNormalisation true $ withReduceDefs (false , dont-reduce) do
      `h1 ← wait-for-type =<< quoteTC h1
      `h2 ← quoteTC h2
      `C ← quoteTC C

      unify goal (“solve” `C (build-expr `h1) (build-expr `h2))

  cat! : {@(tactic cat-worker) p : h1 ≡ h2} → h1 ≡ h2
  cat! {p = p} = p
```
-->

## Demo

As a quick demonstration (and sanity check/future proofing/integration
testing/what have you):

```agda
module _ (C : Precategory o h) where private
  module C = Precategory C
  variable
    A B : C.Ob
    a b c d : C.Hom A B

  test : a C.∘ (b C.∘ (c C.∘ C.id) C.∘ C.id C.∘ (d C.∘ C.id))
       ≡ a C.∘ b C.∘ c C.∘ d
  test = cat! C
```
