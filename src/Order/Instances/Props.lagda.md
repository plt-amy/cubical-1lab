<!--
```agda
open import Cat.Prelude

open import Data.Sum

open import Order.Semilattice.Join
open import Order.Semilattice.Meet
open import Order.Diagram.Bottom
open import Order.Diagram.Join
open import Order.Diagram.Meet
open import Order.Diagram.Glb
open import Order.Diagram.Lub
open import Order.Diagram.Top
open import Order.Base
```
-->

```agda
module Order.Instances.Props where
```

# Propositions

In the [[posetal|poset]] world, what plays the role of the [category of
sets] is the poset of _propositions_ --- our $\le$ relations take values
in propositions, much like the category of sets is where the [$\hom$
functor] takes values.

[category of sets]: Cat.Base.html#the-precategory-of-sets
[$\hom$ functor]: Cat.Functor.Hom.html

Unlike “the” category of sets, which is actually a bunch of categories
(depending on the universe level of the types we're considering), there
is a single poset of propositions --- this is the principle of
[_propositional resizing_], which we assume throughout.

[_propositional resizing_]: 1Lab.Resizing.html

```agda
Props : Poset lzero lzero
Props .Poset.Ob = Ω
Props .Poset._≤_ P Q = ∣ P ∣ → ∣ Q ∣
Props .Poset.≤-thin = hlevel 1
Props .Poset.≤-refl x = x
Props .Poset.≤-trans g f x = f (g x)
Props .Poset.≤-antisym p q = ext (biimp p q)
```

The poset of propositions a top and bottom element, as well as
all meets and joins.

```agda
Props-has-top : Top Props
Props-has-top .Top.top = ⊤Ω
Props-has-top .Top.has-top _ _ = tt

Props-has-bot : Bottom Props
Props-has-bot .Bottom.bot = ⊥Ω
Props-has-bot .Bottom.has-bottom _ ()

Props-has-joins : ∀ P Q → is-join Props P Q (P ∨Ω Q)
Props-has-joins P Q .is-join.l≤join = pure ⊙ inl
Props-has-joins P Q .is-join.r≤join = pure ⊙ inr
Props-has-joins P Q .is-join.least R l r = rec! [ l , r ]

Props-has-meets : ∀ P Q → is-meet Props P Q (P ∧Ω Q)
Props-has-meets P Q .is-meet.meet≤l = fst
Props-has-meets P Q .is-meet.meet≤r = snd
Props-has-meets P Q .is-meet.greatest R l r x = l x , r x

module _ {ℓ} {I : Type ℓ} (Ps : I → Ω) where
  Props-has-glbs : is-glb Props Ps (∀Ω I Ps)
  Props-has-glbs .is-glb.glb≤fam i = rec! (_$ i)
  Props-has-glbs .is-glb.greatest R k x = inc (λ i → k i x)

  Props-has-lubs : is-lub Props Ps (∃Ω I Ps)
  Props-has-lubs .is-lub.fam≤lub i pi = inc (i , pi)
  Props-has-lubs .is-lub.least R k = rec! k
```

<!--
```agda
open is-meet-semilattice
open is-join-semilattice

Props-is-meet-slat : is-meet-semilattice Props
Props-is-meet-slat ._∩_ x y = x ∧Ω y
Props-is-meet-slat .∩-meets = Props-has-meets
Props-is-meet-slat .has-top = Props-has-top

Props-is-join-slat : is-join-semilattice Props
Props-is-join-slat ._∪_ x y    = x ∨Ω y
Props-is-join-slat .∪-joins    = Props-has-joins
Props-is-join-slat .has-bottom = Props-has-bot
```
-->
