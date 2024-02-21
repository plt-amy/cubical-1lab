```agda
module index where
```

# 1lab

A formalised, cross-linked reference resource for cubical methods in
Homotopy Type Theory. Unlike the [@HoTT], the 1lab is not a "linear"
resource: Concepts are presented as a directed graph, with links
indicating _dependencies_. For instance, the statement of the univalence
principle depends on [[universes]], [[identifications|path]] and
[[equivalences]].  In addition to the hyperlinked "web of concepts"
provided by the Agda code, there is a short introduction to homotopy
type theory: **[Start here](1Lab.intro.html)**.

<!--
```agda
open import 1Lab.Type
open import 1Lab.Equiv
open import 1Lab.Path
open import 1Lab.Univalence
open import 1Lab.HLevel
```
-->

```agda
_ : ∀ {ℓ} {A B : Type ℓ} → is-equiv (path→equiv {A = A} {B})
_ = univalence
```

The purpose of the "web of concepts" approach is to let each reader
approach the 1lab at their own pace: If you already know what all of the
code above means, you can click on `univalence`{.Agda} to be taken
directly to the construction of the equivalence --- but if you _don't_,
you can click on other definitions like `is-equiv`{.Agda} and
`path→equiv`{.Agda}, and in turn explore the dependencies of _those_
concepts, and so on.

The 1lab is a community project: we use [GitHub] for source control and
talk on [Discord]. Our purpose is to make cubical methods in homotopy
type theory accessible to, and inclusive of, everyone who is interested,
regardless of cultural background, age, ability, ethnicity, gender
identity, or expression. Correspondingly, interactions in those forums
are governed by the [Contributor Covenant Code of Conduct][cccc]. **We
believe HoTT is for everyone, and are committed to fostering a kind,
inclusive environment.**

[GitHub]: https://github.com/plt-amy/1lab
[Discord]: https://discord.gg/Zp2e8hYsuX
[cccc]: https://github.com/plt-amy/1lab/blob/main/CODE_OF_CONDUCT.md

Mathematics is, fundamentally, a social activity. Correspondingly, we
have a page dedicated to letting authors introduce and talk a bit
themselves and their other work:

```agda
open import Authors
```

Similarly, we maintain this list of related resources which serve as an
introduction to HoTT in general:

* The “canonical” reference is [the HoTT Book], written by a
variety of mathematicians at the IAS Special Year for Univalent
Mathematics, held between 2012-2013 and organised by Steve Awodey,
Thierry Coquand, and the late Vladimir Voevodsky.

  The Book is often referred to on this site - with those words - so if
  you don't know which book "The Book" is, it's the HoTT book! It's
  split into two parts: Type Theory, which introduces the concepts of
  Homotopy Type Theory with no previous knowledge of type theory
  assumed; and Mathematics, which develops some mathematics (homotopy
  theory, category theory, set theory, and real analysis) in this theory.

[the HoTT Book]: https://homotopytypetheory.org/book

  While the 1Lab is not strictly meant to be a companion to, or a
  formalisation of, the HoTT book. But since there is significant
  overlap, one of our pages is simply a list of references to the HoTT
  book and their correspondent on the 1Lab: [HoTT](HoTT.html)

* Prof. Martín Escardó, at Birmingham, has done a great service to the
community by _also_ formalising a great deal of univalent mathematics in
Literate Agda, in his [Introduction to Univalent Foundations of
Mathematics with Agda].

  Prof. Escardó's notes, unlike the 1lab, are done in base Agda, with
  univalence assumed explicitly in the theorems that need it. This is a
  principled decision when the goal is introducing univalent
  mathematics, but it is not practical when the goal is to _practice_
  univalent mathematics in Agda.

  Even still, that document is _much better_ than this site will _ever_
  be as an introduction to the subject! While many of the pages of the
  1lab have introductory _flavour_, it is not meant as an introduction
  to the subject of univalent mathematics.

[Introduction to Univalent Foundations of Mathematics with Agda]: https://www.cs.bham.ac.uk/~mhe/HoTT-UF-in-Agda-Lecture-Notes/HoTT-UF-Agda.html

* Prof. Favonia has kindly uploaded the outline, videos and lecture
notes for their 2020 course on higher-dimensional type theory, which
also serves as an introduction to cubical methods in homotopy type
theory, aimed at graduate students. You can find the course page
[here](https://favonia.org/courses/hdtt2020/), the videos [here on their
YouTube](https://www.youtube.com/playlist?list=PL0OBHndHAAZrGQEkOZGyJu7S7KudAJ8M9),
and the notes [here](https://github.com/favonia/hdtt2020-notes/) (though
heed the warning in the README).

* Another comprehensive, formalised Agda resource is the [agda-unimath]
project, though unlike us (and like prof. Escardó's lecture notes) they
make use of _axiomatic_ HoTT: Univalence is a postulate, and thus does
not have computational content.

  Regardless, they have formalised a great deal of "ordinary"
  mathematics in the univalent context: elementary number theory, group
  theory and combinatorics being the most prominent projects.

[agda-unimath]: https://unimath.github.io/agda-unimath/

## Technology

The 1Lab uses [Julia Mono](https://juliamono.netlify.app/) as its
monospace typeface. Julia Mono is licensed under the SIL Open Font
License, v1.1, a copy of which can be found
[here](/static/licenses/LICENSE.JuliaMono). As the sans-serif typeface, we
use the [Inria Sans] webfont, and as a serif typeface, [EB Garamond].
These fonts are both open-source, though rather than rehosting them, we
use them from Google Fonts.

[Inria Sans]: https://fonts.google.com/specimen/Inria+Sans
[EB Garamond]: https://fonts.google.com/specimen/EB+Garamond

Mathematics is rendered using [KaTeX](https://katex.org), and as so, the
1Lab redistributes KaTeX's fonts and stylesheets, even though the
rendering is done entirely at build-time. KaTeX is licensed under the
MIT License, a copy of which can be found
[here](/static/licenses/LICENSE.KaTeX).

Our favicon is Noto Emoji's ice cube (cubical type theory - get it?),
codepoint U+1F9CA. This is the only image from Noto we redistribute.
Noto fonts are licensed under the Apache 2.0 License, a copy of which
can be found [here](/static/licenses/LICENSE.Noto).

Commutative diagrams appearing in body text are created using
[quiver](https://q.uiver.app), and rendered to SVG using a combination of
[LaTeX](https://tug.org/texlive/) and
[pdftocairo](https://poppler.freedesktop.org/), part of the Poppler
project. No part of these projects is redistributed.

And, of course, the formalisation would not be possible without
[Agda](https://github.com/agda/agda).

# Type theory

::: warning
Most of the modules in the 1Lab assume a baseline knowledge of type
theory. For this, [**read the introduction here**](1Lab.intro.html).
:::

The first things to be explained are the foundational constructions in
(cubical) type theory - things like types themselves, [[universes]],
[[paths]], [[equivalences]], [[glueing]] and the [[univalence]] "axiom".
These are developed under the `1Lab` namespace. Start here:

```agda
-- All of these module names are links you can click!

open import 1Lab.Type -- Universes
open import 1Lab.Type.Pointed -- Pointed types

open import 1Lab.Path -- Path types
open import 1Lab.Path.Groupoid  -- Groupoid structure of types
open import 1Lab.Path.Reasoning -- Combinators for reasoning with path composition
open import 1Lab.Path.IdentitySystem -- Families R for which R(x,y) ≃ (x ≡ y)
open import 1Lab.Path.IdentitySystem.Strict -- Identity systems on sets

open import 1Lab.Equiv -- “Contractible fibres” equivalences
open import 1Lab.Equiv.Biinv -- Biinvertible maps
open import 1Lab.Equiv.FromPath -- Transport is an equivalence, cubically
open import 1Lab.Function.Embedding -- Embeddings
open import 1Lab.Equiv.Fibrewise -- Fibrewise equivalences
open import 1Lab.Equiv.HalfAdjoint -- Half-adjoint equivalences

open import 1Lab.HLevel -- h-levels
open import 1Lab.HLevel.Retracts -- Closure of h-levels under retractions/isos
open import 1Lab.HLevel.Universe -- The type of n-types is a (n+1)-type

open import 1Lab.Univalence -- Equivalence is equivalent to identification
open import 1Lab.Univalence.SIP -- Univalence + preservation of structure
open import 1Lab.Univalence.SIP.Auto -- Derive is-univalent for families of types

open import 1Lab.Type.Pi -- Properties of dependent products
open import 1Lab.Type.Sigma -- Properties of dependent coproducts

open import 1Lab.HIT.Truncation -- Propositional truncation

open import 1Lab.Classical -- Classical logic

open import 1Lab.Counterexamples.IsIso -- Counterexample: is-iso is not a prop
open import 1Lab.Counterexamples.Russell -- Counterexample: Russell's paradox
open import 1Lab.Counterexamples.Sigma -- Counterexample: Sigma is not prop
open import 1Lab.Counterexamples.GlobalChoice -- Counterexample: global choice is inconsistent with univalence
```

## Data types

The `Data` namespace contains definitions of oft-used data types, which
are fundamental to the rest of the development but not “basic type
theory”. These modules contain (or re-export) the types themselves,
useful operations on them, characterisation of their path spaces, etc.

The natural numbers have a lot of associated theory (number theory), so
there are a lot of child modules of `Data.Nat`{.Agda}:

```agda
open import Data.Nat  -- The natural numbers
open import Data.Nat.Solver
-- Commutative semiring solver for equations in Nat

open import Data.Nat.DivMod -- Euclidean division
open import Data.Nat.Divisible -- Divisibility
open import Data.Nat.Divisible.GCD
-- The greatest common divisor, Euclid's algorithm

open import Data.Nat.Order -- Properties of ≤
open import Data.Nat.Properties -- Arithmetic properties
```

We also have a theory of finite sets:

```agda
open import Data.Fin
open import Data.Fin.Base -- The standard finite sets
open import Data.Fin.Finite -- Finiteness
open import Data.Fin.Closure -- Closure properties of finiteness
open import Data.Fin.Properties -- Properties of finite sets
```

Of similar importance is the type of integers:

```agda
open import Data.Int.HIT              -- The integers (as a higher inductive type!)
open import Data.Int.Base             -- Inductively-defined integers
open import Data.Int.Order            -- ≤ on the integers
open import Data.Int.Universal        -- A universal property of the integers
open import Data.Int.Properties       -- Algebra on the integers
```

General constructions on sets:

```agda
open import Data.Set.Truncation -- Set truncations
open import Data.Set.Surjection -- Surjections of sets
open import Data.Set.Coequaliser -- Set coequalisers
```

Well-founded relations, well-founded trees and well-founded induction:

```agda
open import Data.Wellfounded.W -- Trees
open import Data.Wellfounded.Base -- Relations
open import Data.Wellfounded.Properties -- Properties of well-founded relations
```

And general data types:

```agda
open import Data.Sum  -- Coproduct types
open import Data.Dec  -- Decisions and decidable types
open import Data.Bool -- The booleans
open import Data.List -- Finite lists
open import Data.Maybe -- The Maybe type
```

We also consider "data types" to encompass properties of properties, or,
more generally, predicates:

```agda
open import Data.Power -- Power sets
open import Data.Power.Complemented -- Complemented or decidable subobjects
```

# Category theory

In addition to providing a framework for the synthetic study of higher
groupoids, HoTT also provides a natural place to develop constructive
category theory, while still being compatible with classicality
principles like the axiom of choice and/or the law of excluded middle.
Here, we do not assume any classicality principles.

## Basics

The main modules in the `Cat` namespace provide the foundation for the
rest of the development, defining basic constructions like precategories
themselves, functors, natural transformations, etc.

```agda
open import Cat.Base -- Precategories, functors, natural transformations
open import Cat.Solver -- Automatic solver for associativity problems
open import Cat.Morphism  -- Important classes of morphisms
open import Cat.Reasoning -- Categorical reasoning combinators
open import Cat.Groupoid -- Groupoids
open import Cat.Connected -- Connected categories
open import Cat.Finite -- Finite categories
```

### Diagrams

For convenience, we define a plethora of "concrete" universal diagrams,
unpacking their definitions as limits or colimits. These are simpler to
work with since they provide the relevant data with fewer layers of
indirection.

```agda
open import Cat.Diagram.Congruence -- Internal equivalence relations

-- Colimits:
open import Cat.Diagram.Initial -- Initial objects
open import Cat.Diagram.Pushout -- Pushouts
open import Cat.Diagram.Coproduct -- Binary coproducts
open import Cat.Diagram.Coproduct.Copower -- Copowers
open import Cat.Diagram.Coequaliser -- Coequalisers
open import Cat.Diagram.Colimit.Base -- Conical colimits
open import Cat.Diagram.Colimit.Finite -- Finite colimits
open import Cat.Diagram.Coproduct.Indexed -- Indexed coproducts
open import Cat.Diagram.Coequaliser.RegularEpi -- Regular epimorphisms

-- Coends:
open import Cat.Diagram.Coend -- Coends
open import Cat.Diagram.Coend.Sets -- Concrete computation of coends in sets
open import Cat.Diagram.Coend.Formula -- A formula for computing coends

open import Cat.Diagram.Duals -- Dualisation of co/limits
open import Cat.Diagram.Image -- Image factorisations
open import Cat.Diagram.Sieve -- Subobjects of a Hom-functor
open import Cat.Diagram.Idempotent -- Idempotent morphisms

-- Limits
open import Cat.Diagram.Product -- Binary products
open import Cat.Diagram.Product.Finite -- n-Ary products
open import Cat.Diagram.Product.Solver
-- Automatic solving of equations in a Cartesian monoidal category

open import Cat.Diagram.Pullback -- Fibred products
open import Cat.Diagram.Terminal -- Terminal objects
open import Cat.Diagram.Equaliser -- Equalisers
open import Cat.Diagram.Limit.Base -- Conical limits
open import Cat.Diagram.Limit.Finite
open import Cat.Diagram.Limit.Product
open import Cat.Diagram.Limit.Pullback
open import Cat.Diagram.Limit.Equaliser
open import Cat.Diagram.Product.Indexed
open import Cat.Diagram.Equaliser.Kernel -- Kernels
open import Cat.Diagram.Pullback.Properties -- Properties of fibred products
open import Cat.Diagram.Equaliser.RegularMono -- Regular monomorphisms

open import Cat.Diagram.Monad -- Monads
open import Cat.Diagram.Monad.Limits -- Limits in Eilenberg-Moore categories
open import Cat.Diagram.Monad.Codensity -- Codensity monads
open import Cat.Diagram.Monad.Extension -- Extension systems
open import Cat.Diagram.Monad.Relative -- Relative extension systems

open import Cat.Diagram.Zero -- Zero objects
```

### Interesting morphisms

There are a number of properties, constructions and classes of morphisms
we can construct in _any_ category.

```agda
open import Cat.Morphism.StrongEpi -- Strong epimorphisms
open import Cat.Morphism.Orthogonal -- Orthogonality
open import Cat.Morphism.Factorisation -- Factorisation systems
open import Cat.Morphism.Duality  -- Duality of morphism classes
```

## Functors

This namespace has definitions of properties functors can have, utility
modules for working with functors, the definition of full subcategories,
and adjoint functors.

```agda
open import Cat.Functor.Base -- Compendium of functor properties
open import Cat.Functor.Dense -- Dense functors
open import Cat.Functor.Final -- Final functors
open import Cat.Functor.Slice -- Extending functors to slice categories
open import Cat.Functor.Pullback -- Base change, dependent sum, Σf ⊣ f*
open import Cat.Functor.Amnestic -- Functors which reflect univalence
open import Cat.Functor.Bifunctor -- Functors out of product categories
open import Cat.Functor.Conservative -- Functors which reflect isomorphisms
open import Cat.Functor.FullSubcategory -- Full subcategories
open import Cat.Functor.WideSubcategory -- Wide subcategories
open import Cat.Functor.Subcategory -- Subcategories, generally
```

Helpers for working with functions in equational reasoning:

```agda
open import Cat.Functor.Reasoning
open import Cat.Functor.Reasoning.FullyFaithful
```

About equivalences of (pre)categories:

```agda
open import Cat.Functor.Equivalence -- Equivalences of (pre)categories
open import Cat.Functor.Equivalence.Path
  -- Categories are identical by their equivalences
open import Cat.Functor.Equivalence.Complete -- Equivalences preserve completeness
```

About [[adjoint functors]], and their associated monads:

```agda
open import Cat.Diagram.Monad -- Definition of monads
open import Cat.Functor.Adjoint -- Unit-counit adjunctions and universal arrows
open import Cat.Functor.Adjoint.Hom -- Adjoints in terms of Hom-isomorphisms
open import Cat.Functor.Adjoint.Representable -- Adjoints in terms of representables
open import Cat.Functor.Adjoint.Monad -- Monad from an adjunction
open import Cat.Functor.Adjoint.Unique -- Uniqueness of adjoints
open import Cat.Functor.Adjoint.Monadic -- Monadic adjunctions
open import Cat.Functor.Adjoint.Compose -- Adjunctions compose
open import Cat.Functor.Adjoint.Continuous -- Right adjoints preserve limits
open import Cat.Functor.Adjoint.Reflective -- Reflective subcategories
open import Cat.Functor.Adjoint.Mate -- Mates of adjoints
```

Monadicity theorems:

```agda
open import Cat.Functor.Monadic.Beck  -- Beck's coequalisers
open import Cat.Functor.Monadic.Crude -- The crude monadicity theorem
```

About [[Kan extensions]]:

```agda
open import Cat.Functor.Kan.Base -- Kan extensions
open import Cat.Functor.Kan.Duality -- Left and right extensions are dual
open import Cat.Functor.Kan.Nerve -- The nerve/realisation adjunction
open import Cat.Functor.Kan.Global -- Global Kan extensions
open import Cat.Functor.Kan.Adjoint -- Adjoints are Kan extensions
open import Cat.Functor.Kan.Pointwise -- Pointwise Kan extensions
open import Cat.Functor.Kan.Unique -- Uniqueness of Kan extensions
open import Cat.Functor.Kan.Representable -- Kan extensions as Hom isomorphisms
open import Cat.Functor.Kan.Isbell -- Isbell duality between Presheaves and Copresheaves
```

Properties of Hom-functors, and (direct) consequences of the Yoneda
lemma:

```agda
open import Cat.Functor.Hom -- Hom functor, Yoneda embedding
open import Cat.Functor.Hom.Cocompletion -- Universal property of PSh(C)
open import Cat.Functor.Hom.Coyoneda -- The Coyoneda lemma
open import Cat.Functor.Hom.Representable -- Representable functors
open import Cat.Functor.Hom.Duality -- Duality of Hom functors

open import Cat.Functor.Hom.Displayed
  -- Hom functors of displayed categories
```

## Univalent categories

In HoTT/UF, the word "category" is reserved for the precategories (what
the rest of the world refers to as just "category") in which isomorphic
objects are indistinguishable, i.e. the categories which satisfy a
version of the [[univalence axiom]]. Sometimes we also refer to these as
"[[univalent categories]]" to make the distinction clear.

```agda
open import Cat.Univalent -- Basic properties of categories
open import Cat.Univalent.Rezk -- Free category on a precategory
open import Cat.Univalent.Rezk.Universal
  -- Universal property of the Rezk completion
open import Cat.Univalent.Instances.Algebra
  -- Eilenberg-Moore categories preserve univalence
```

## Strict categories

In general, precategories do not have a set of objects. We call categories
that do **strict**.

```agda
open import Cat.Strict -- Categories with a set of objects.
open import Cat.Skeletal -- Categories where isomorphisms are automorphisms.
open import Cat.Gaunt -- Strict univalent categories.
```

Properties, constructions, and the [[category of strict categories]]:

```agda
-- Strict categories
open import Cat.Instances.StrictCat
open import Cat.Instances.StrictCat.Cohesive
  -- ^ Strict category structure is a sort of "spatial" structure on a
  -- set

open import Cat.Instances.Free
-- Free strict categories on a directed graph

open import Cat.Instances.FinSet
-- Skeleton of the category of finite sets

open import Cat.Instances.Simplex
-- Skeleton of the simplex category

open import Cat.Instances.Discrete -- Discrete categories
open import Cat.Instances.Delooping -- Delooping a monoid to give a category
```

## Category instances

Category "instances" are constructions of, and proofs associated to, the
construction of actual categories, rather than reasoning about
categories in the abstract. We begin with some assorted constructions:

```agda
open import Cat.Instances.Elements -- Category of elements of a presheaf

open import Cat.Instances.Karoubi
-- Completion of a category under splitting of idempotents

open import Cat.Instances.Twisted
-- The twisted arrow category (used in computing co/ends)

open import Cat.Instances.Lift
-- Lifting a category to higher universes

open import Cat.Instances.Product -- Product categories

open import Cat.Instances.Localisation
-- Freely inverting a class of morphisms in a category.

open import Cat.Instances.Core
-- The core of a category
```

The construction and properties of functor categories:

```agda
-- Functor categories:
open import Cat.Functor.Base
open import Cat.Functor.Compose
-- Composition of functors is functorial (also whiskering natural
-- transformations)
open import Cat.Instances.Functor.Limits -- Co/limits in functor categories
open import Cat.Instances.Functor.Duality -- 2-cell duality in Cat
```

The internal versions of functor categories:
```agda
-- Internal functor categories:
open import Cat.Instances.InternalFunctor
open import Cat.Instances.InternalFunctor.Compose
-- Composition of internal functors is functorial (also whiskering internal
-- natural transformations)
open import Cat.Instances.OuterFunctor
-- The category of functors from an internal category to its base.
```

Properties of the category of sets:

```agda
-- The category of sets:
open import Cat.Instances.Sets -- is univalent
open import Cat.Instances.Sets.Complete -- is complete
open import Cat.Instances.Sets.Cocomplete -- is cocomplete, with disjoint coproducts
open import Cat.Instances.Sets.Congruences -- has effective congruences
open import Cat.Instances.Sets.CartesianClosed -- and is locally cartesian closed
open import Cat.Instances.Sets.Counterexamples.SelfDual -- and is not self-dual
```

The category of polynomial functors:

```agda
open import Cat.Instances.Poly
```

A few concrete, tiny categories are classed as "diagram shapes":

```agda
-- Diagram shapes:
open import Cat.Instances.Shape.Join
open import Cat.Instances.Shape.Cospan
open import Cat.Instances.Shape.Interval
open import Cat.Instances.Shape.Parallel
open import Cat.Instances.Shape.Terminal
open import Cat.Instances.Shape.Isomorphism -- The walking isomorphism
```

Slice categories and comma objects:

```agda
open import Cat.Instances.Comma -- Comma categories
open import Cat.Instances.Slice -- Slice categories
open import Cat.Instances.Slice.Presheaf -- PSh(C)/X ≅ PSh(∫ X)
open import Cat.Instances.Comma.Univalent
```

## Cartesian closed categories

A [[Cartesian closed]] category, or CCC for short, is one that has
_internalisations_ for all its $\hom$-sets: [[exponential objects]]. Put
another way, a CCC interprets the [[simply-typed lambda calculus]]. Also
of interest are the [[*locally* Cartesian closed categories]], where we
also have an interpretation for _dependent product_ types.

```agda
open import Cat.Diagram.Exponential
open import Cat.CartesianClosed.Lambda
open import Cat.CartesianClosed.Locally
```


## Allegories

Allegories are abstractions of the nice properties that the category of
relations enjoys. They are, strictly speaking, _bicategories_, but since
they are locally posets, we have a definition of allegory free of all
the extra coherence that is necessary for specifying a bicategory.

```agda
open import Cat.Allegory.Base -- The definition
open import Cat.Allegory.Maps -- Functional relations in an allegory
open import Cat.Allegory.Morphism -- Morphisms in allegories
open import Cat.Allegory.Reasoning -- Reasoning combinators
```

## Restriction categories

Restriction categories axiomatize categories of partial maps by adding
n restriction operation $(-)\downarrow : \cC(X,Y) \to \cC(X,X)$ that
takes a morphism $f$ to a subobject of the identity morphism that is
defined precisely when $f$ is.

```agda
open import Cat.Restriction.Base
  -- The definition
open import Cat.Restriction.Functor
  -- Functors between restriction categories
open import Cat.Restriction.Reasoning
  -- Reasoning combinators and morphism classes
open import Cat.Restriction.Total
  -- Categories of total maps
```

```agda
open import Cat.Restriction.Instances.Allegory
 -- Restriction structures on partial maps of an allegory.
```


## Displayed categories

A category displayed over $\cB$ is a particular concrete presentation
of the bicategorical slice $\Cat/\cB$; that is, it is a better way of
presenting the data of a category $\cE$ and a functor $\cE \to
\cB$.

In addition to the _extensive_ use of [[displayed categories]] to model
"pre-indexing" in the 1Lab, we also contain an in-progress formalisation
of [Foundations of Relative Category Theory][frct].

[frct]: https://www.jonmsterling.com/frct-003I.xml

```agda
open import Cat.Displayed.Base -- Displayed categories

open import Cat.Displayed.Total -- Total category of a displayed category
open import Cat.Displayed.Total.Free -- Free objects in a total category

open import Cat.Displayed.Total.Op -- Total opposite categories

open import Cat.Displayed.Fibre -- Fibre categories of a displayed category

open import Cat.Displayed.Univalence -- Univalence for displayed categories
open import Cat.Displayed.Univalence.Thin -- The structure identity principle

open import Cat.Displayed.Reasoning
  -- Reasoning combinators for displayed categories

open import Cat.Displayed.Morphism
  -- Important classes of morphisms in displayed categories

open import Cat.Displayed.Morphism.Duality
  -- Duality of morphism classes in displayed categories

open import Cat.Displayed.Instances.Elements
-- The category of elements of a presheaf, instantiated as being
-- displayed over the domain.

open import Cat.Displayed.Instances.TotalProduct
-- The total product of two displayed categories.

open import Cat.Displayed.Composition
  -- Composition of displayed categories
```

We can also investigate how displayed categories relate to other
displayed categories (over the same base, or over different bases), and
their higher groupoid structure:

```agda
open import Cat.Displayed.Path
open import Cat.Displayed.Functor
open import Cat.Displayed.Adjoint
```

### Cartesian fibrations

In the land of 1-categories, the notion of "indexed families of objects"
is accurately captured by [slice categories]. But when we're talking
about the 2-category $\Cat$, plain functors won't do. In terms of
displayed categories, we need to talk about _[[Cartesian fibrations]]_
instead. These satisfy a property analogous to the existence of
pullbacks, and they are precisely those which correspond to families
$\cB \to \Cat$.

[slice categories]: Cat.Instances.Slice.html

```agda
open import Cat.Displayed.Cartesian
 -- Cartesian lifts, cartesian fibrations

open import Cat.Displayed.Cartesian.Weak
 -- Weak cartesian morphisms

open import Cat.Displayed.Cartesian.Street
-- Street's fibrations

open import Cat.Displayed.Cartesian.Discrete
-- Discrete fibrations are presheaves

open import Cat.Displayed.Cartesian.Right
-- Fibrations in groupoids

open import Cat.Displayed.Cartesian.Indexing
-- Fibrations have pseudofunctorial reindexing.
```

Not to dwell on a vacuous concept, we also have constructions of
Cartesian fibrations:

```agda
open import Cat.Displayed.Instances.Slice -- Canonical self-indexing
open import Cat.Displayed.Instances.Subobjects -- Fibration of subobjects
open import Cat.Displayed.Instances.Family -- Family fibration
open import Cat.Displayed.Instances.DisplayedFamilies
-- Families internal to a fibration.
open import Cat.Displayed.Instances.Pullback
  -- Pullback of a displayed category by a functor
open import Cat.Displayed.Instances.Scone
-- We can consider *scones* over a category C with a terminal object as
-- forming a displayed category over C. Moreover, it's a Cartesian
-- fibration by construction.
open import Cat.Displayed.Instances.Trivial
-- Any category can be displayed over the terminal category.
open import Cat.Displayed.Instances.Lifting
-- Liftings of functors along a fibration
open import Cat.Displayed.Instances.Diagrams
-- The fibration of diagrams
open import Cat.Displayed.Instances.Objects
-- The fibration of objects.
open import Cat.Displayed.Instances.Externalisation
-- Internal categories as fibrations.
```

### Cocartesian fibrations

```agda
open import Cat.Displayed.Cocartesian
  -- Cocartesian lifts, opfibrations

open import Cat.Displayed.Cocartesian.Indexing
  -- Opfibrations have covariant opreindexing

open import Cat.Displayed.Cocartesian.Weak
  -- Weak cocartesian morphisms
```

### Bifibrations

```agda
open import Cat.Displayed.Bifibration
  -- Bifibrations, adjoints to base change
open import Cat.Displayed.Instances.Chaotic
-- The bifibration associated with the projection functor
-- $\cB \times \cJ \to \cB$.
open import Cat.Displayed.Instances.Identity
-- The bifibration associated with the identity functor.
```

### Structures in fibrations

```agda
open import Cat.Displayed.InternalSum
-- The fibred equivalent of sigma types and existential quantifiers
open import Cat.Displayed.GenericObject
-- Generic objects in fibrations.
```

### Logical structure of fibrations

Fibrations serve as an excellent foundation for exploring various
logical and type-theoretic phenomena.

```agda
open import Cat.Displayed.Comprehension
-- A categorical model of context extension.
open import Cat.Displayed.Comprehension.Coproduct
-- Coproducts in comprehension categories
open import Cat.Displayed.Comprehension.Coproduct.Strong
-- Coproducts with a stronger elimination principle
open import Cat.Displayed.Comprehension.Coproduct.VeryStrong
-- Coproducts with a very strong elimination principle
```


## Internal categories

The theory of internal categories. Internal category theory generalizes
[[strict category]] theory by replacing the ambient category
$\thecat{Sets}$ with an arbitrary category $\cC$ with pullbacks.

```agda
open import Cat.Internal.Base
-- Internal categories, internal functors, and internal natural
-- transformations.

open import Cat.Internal.Morphism
-- Internal monos, epis, and isos.

open import Cat.Internal.Reasoning
-- Reasoning combinators for internal categories.

open import Cat.Internal.Opposite
-- The opposite of an internal category.

open import Cat.Internal.Functor.Outer
-- Internal functors from an internal category to its base.

open import Cat.Internal.Sets
-- Categories internal to sets are strict categories.
```

### Examples of internal categories

```agda
open import Cat.Internal.Instances.Discrete
-- Discrete internal categories.

open import Cat.Internal.Instances.Congruence
-- Internal equivalence relations are internal categories.
```

## Bicategories

The theory of bicategories, as isolated objects. Note that a
comprehensive study of how bicategories interact with other bicategories
is a **tri**categorical problem!

```agda
open import Cat.Bi.Base
  -- Prebicategories, lax functors, pseudofunctors, lax transformations,
  -- pseudonatural transformations, modifications, and the bicategory of
  -- categories.
open import Cat.Bi.Instances.Spans
  -- The prebicategory of spans in a precategory with pullbacks
open import Cat.Bi.Instances.Discrete
  -- The locally discrete prebicategory associated to a precategory
open import Cat.Bi.Instances.InternalCats
  -- The prebicategory of categories internal to a fixed base category.
```

### Diagrams in bicategories

```agda
open import Cat.Bi.Diagram.Monad -- Monads in a bicategory
open import Cat.Bi.Diagram.Monad.Spans  -- Internal categories as monads in Span(C)
open import Cat.Bi.Diagram.Adjunction -- Adjunction diagrams in a bicategory
```

## Monoidal categories

In addition to general bicategories, we also have bicategories with one
object.

```agda
open import Cat.Monoidal.Base
open import Cat.Monoidal.Diagram.Monoid
open import Cat.Monoidal.Instances.Cartesian

open import Cat.Monoidal.Diagram.Monoid.Representable
-- Internal monoids, representability, and the internal language of a category.
```

## Homological algebra

The theory of abelian (and Ab-enriched) categories in general, and
specific constructions of abelian categories.

```agda
open import Cat.Abelian.Base -- Definition of the tower of Ab-enriched categories
open import Cat.Abelian.Endo -- Endomorphism rings
open import Cat.Abelian.Images  -- Image factorisations in abelian categories
open import Cat.Abelian.Limits  -- Finite biproducts in abelian categories
open import Cat.Abelian.Functor -- Ab-enriched functors

open import Cat.Abelian.Instances.Ab
  -- The category of abelian groups
open import Cat.Abelian.Instances.Functor
  -- Ab-enriched functor categories
```

## Topos theory

Grothendieck topos theory developed constructively.

```agda
open import Topoi.Base -- Topoi, properties of topoi, geometric morphisms
open import Topoi.Reasoning  -- Exactness properties of topoi (cont'd), reasoning
open import Topoi.Classifying.Diaconescu
-- ^ Presheaf topoi classify flat functors on their site
```

# Order theory

Order theory is, to the category theorist, the study of 0-categories:
Those for which we have a (-1)-groupoid of morphisms between any two
objects, i.e., those for which rather than having $\hom$-sets, we have a
$x \le y$ relation.

```agda
open import Order.Base -- Definitions
open import Order.Cat  -- Posets generate categories
open import Order.Reasoning -- Nice syntax for posets
open import Order.Displayed -- Displayed posets
```

For readability, we have diagrams in orders separate from diagrams in
their generated categories:

```agda
open import Order.Diagram.Glb
open import Order.Diagram.Lub
open import Order.Diagram.Fixpoint -- Least and Greatest fixpoints
```

Some order-theoretic structures are equivalently presented as algebraic
structures: these are the lattices and related structures.

```agda
open import Order.Frame
open import Order.Lattice

open import Order.Semilattice.Join
open import Order.Semilattice.Meet
open import Order.Semilattice.Free
```

Examples of actual orders:

```agda
open import Order.Instances.Discrete -- Discrete posets
open import Order.Instances.Props -- Ω
open import Order.Instances.Lower -- Lower sets

open import Order.Instances.Pointwise -- The pointwise ordering on A→B
open import Order.Instances.Pointwise.Diagrams
```

## Domain theory

Domain theory is the study of posets that are complete
under various classes of least upper bounds. These posets are used
to model notions of partiality, which makes them extremely useful
in the search for semantics of various programming languages.

```agda
open import Order.DCPO -- Directed-complete partial orders
open import Order.DCPO.Pointed -- Pointed directed-complete partial orders
open import Order.DCPO.Free -- Free DCPOs and free pointed DCPOs
```

# Logic

```agda
open import Logic.Propositional.Classical
-- Classical logic, soundness, completeness
open import Logic.Propositional.Classical.CNF
-- Conjunctive normal forms
open import Logic.Propositional.Classical.SAT
-- DPLL SAT solver
open import Logic.Propositional.Classical.Compactness
-- Compactness of classical propositional logic is a constructive taboo
```

# Algebra

```agda
open import Algebra.Magma -- Binary operations
open import Algebra.Magma.Unital -- Operations with two-sided units
open import Algebra.Magma.Unital.EckmannHilton -- The Eckmann-Hilton argument

open import Algebra.Semigroup -- Semigroups (associative magmas)

open import Algebra.Monoid -- Monoids as unital semigroups
open import Algebra.Monoid.Category -- The category of monoids
```

## Group theory

```agda
open import Algebra.Group -- Groups as monoids with inverses
open import Algebra.Group.NAry -- NAry sums on groups
open import Algebra.Group.Free -- Free groups
open import Algebra.Group.Action -- Group actions
open import Algebra.Group.Cayley -- Cayley's theorem
open import Algebra.Group.Cat.Base -- The category of groups
open import Algebra.Group.Cat.Monadic -- ... is monadic over Sets
open import Algebra.Group.Cat.FinitelyComplete -- Finite limits in Groups
open import Algebra.Group.Subgroup -- Subgroups, images and kernels
open import Algebra.Group.Concrete -- Concrete groups (pointed connected groupoids)
open import Algebra.Group.Concrete.Abelian -- Concrete abelian groups

open import Algebra.Group.Homotopy -- Homotopy groups
open import Algebra.Group.Homotopy.BAut
  -- Delooping groupoids of automorphism groups

open import Algebra.Group.Ab -- Abelian groups, and the category Ab
open import Algebra.Group.Ab.Sum  -- Direct sum abelian groups
open import Algebra.Group.Ab.Free -- The free abelian group on a group
```

## Ring theory

```agda
open import Algebra.Ring -- Rings
open import Algebra.Ring.Ideal -- Ideals in commutative rings
open import Algebra.Ring.Module -- Modules over a commutative ring
open import Algebra.Ring.Quotient -- Quotient rings
open import Algebra.Ring.Cat.Initial -- ℤ is the initial ring
open import Algebra.Ring.Commutative -- Commutative rings

open import Algebra.Ring.Module.Vec -- Finite direct sums of R as an R-module
open import Algebra.Ring.Module.Free -- Free R-modules as a HIT
open import Algebra.Ring.Module.Category -- The bifibration of Mod over Ring
```

# Homotopy theory

Synthetic homotopy theory is the name given to studying
$\infty$-groupoids in their own terms, i.e., the application of homotopy type
theory to computing homotopy invariants of spaces.

```agda
open import Homotopy.Base -- Basic definitions
open import Homotopy.Connectedness -- Connected types
open import Homotopy.Connectedness.Automation -- Automation for connectedness

open import Homotopy.Space.Suspension -- Suspensions
open import Homotopy.Space.Circle -- The circle
open import Homotopy.Space.Sphere -- The n-spheres
open import Homotopy.Space.Sinfty -- The ∞-sphere
open import Homotopy.Space.Torus -- The torus
```

<!-- Mastodon author links: !-->
<div style="display: none;">
<a rel="me" href="https://types.pl/@amy"></a>
<a rel="me" href="https://types.pl/@totbwf"></a>
</div>
