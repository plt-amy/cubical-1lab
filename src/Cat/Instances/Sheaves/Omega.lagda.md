<!--
```agda
open import Cat.Diagram.Sieve
open import Cat.Site.Closure
open import Cat.Site.Base
open import Cat.Prelude

import Cat.Reasoning as Cat
```
-->

```agda
module Cat.Instances.Sheaves.Omega {ℓ} {C : Precategory ℓ ℓ} (J : Coverage C ℓ) where
```

# Closed sieves and the subobject classifier {defines="subobject-classifier-sheaf"}

<!--
```agda
open Functor
open Cat C

open Coverage J using (Membership-covers ; Sem-covers)
```
-->

The category of [[sheaves]] $\Sh(\cC, J)$ on a small [[site]] $\cC$ is a
*topos*, which means that --- in addition to finite limits and
exponential objects --- it has a *subobject classifier*, a sheaf which
plays the role of the "universe of propositions". We can construct the
sheaf $\Omega_J$ explicitly, as the sheaf of **$J$-closed sieves**.

A sieve is $J$-closed if it contains every morphism it covers. This
notion is evidently closed under pullback, so the closed sieves form a
presheaf on $\cC$.

```agda
is-closed : ∀ {U} → Sieve C U → Type ℓ
is-closed {U} S = ∀ {V} (h : Hom V U) → J ∋ pullback h S → h ∈ S

abstract
  is-closed-pullback
    : ∀ {U V} (f : Hom V U) (S : Sieve C U)
    → is-closed S → is-closed (pullback f S)
  is-closed-pullback f S c h p = c (f ∘ h) (subst (J ∋_) (sym pullback-∘) p)
```

<!--
```agda
private instance
  Extensional-closed-sieve : ∀ {ℓr U} ⦃ _ : Extensional (Sieve C U) ℓr ⦄ → Extensional (Σ[ R ∈ Sieve C U ] is-closed R) ℓr
  Extensional-closed-sieve ⦃ e ⦄ = injection→extensional! Σ-prop-path! e
```
-->

```agda
ΩJ : Sheaf J ℓ
ΩJ .fst = pre where
  pre : Functor (C ^op) (Sets ℓ)
  pre .F₀ U = el! (Σ[ R ∈ Sieve C U ] is-closed R)
  pre .F₁ f (R , c) = pullback f R , is-closed-pullback f R c
  pre .F-id    = funext λ _ → Σ-prop-path! pullback-id
  pre .F-∘ f g = funext λ _ → Σ-prop-path! pullback-∘
```

It remains to show that this is a sheaf. We start by showing that it is
[[separated|separated presheaf]]. Suppose we have two closed sieves $S$,
$T$ which agree everywhere on some $R \in J(U)$. We want to show $S =
T$, so fix some $h : V \to U$; we'll show $h \in S$ iff $h \in T$.

```agda
ΩJ .snd = from-is-separated sep mk where
```

It appears that we don't know much about how $S$ and $T$ behave outside
of their agreement on $R$, but knowing that they're closed will be
enough to show that they agree everywhere. First, let's codify that they
actually agree on their intersection with $R$:

```agda
  sep : is-separated J (ΩJ .fst)
  sep {U} R {S , cS} {T , cT} α = ext λ {V} h →
    let
      rem₁ : S ∩S ⟦ R ⟧ ≡ T ∩S ⟦ R ⟧
      rem₁ = ext λ {V} h → biimp
        (λ (h∈S , h∈R) → cT h (subst (J ∋_) (ap fst (α h h∈R)) (max (S .closed h∈S id))) , h∈R)
        (λ (h∈T , h∈R) → cS h (subst (J ∋_) (ap fst (sym (α h h∈R))) (max (T .closed h∈T id))) , h∈R)
```

Then, assuming w.l.o.g. that $h \in S$, we know that the pullback $h^*(S
\cap R)$ is a covering sieve. And since $h^*(T)$ is a subsieve of
$h^*(S \cap R) = h^*(T \cap R)$, we conclude that if $h \in S$, then
$h^*(T)$ is $J$-covering; and since $T$ is closed, this implies also
that $h \in T$.

```agda
      rem₂ : h ∈ S → J ∋ pullback h (S ∩S ⟦ R ⟧)
      rem₂ h∈S = local (pull h (inc R)) λ f hf∈R → max
        ( S .closed h∈S (f ∘ id)
        , subst (_∈ R) (ap (h ∘_) (intror refl)) hf∈R
        )

      rem₂' : h ∈ S → J ∋ pullback h T
      rem₂' h∈S = incl (λ _ → fst) (subst (J ∋_) (ap (pullback h) rem₁) (rem₂ h∈S))
```

We omit the symmetric converse for brevity.

<!--
```agda
      rem₃ : h ∈ T → J ∋ pullback h S
      rem₃ ht = incl (λ _ → fst) (subst (J ∋_) (ap (pullback h) (sym rem₁))
        (local (pull h (inc R)) λ f rfh → max (T .closed ht (f ∘ id) , subst (_∈ R) (ap (h ∘_) (intror refl)) rfh)))
```
-->

```agda
    in biimp (λ h∈S → cT h (rem₂' h∈S)) (λ h∈T → cS h (rem₃ h∈T))
```

Now we have to show that a family $S$ of compatible closed sieves over a
$J(U)$-covering sieve $R$ can be uniquely patched to a closed sieve on
$U$. This is the sieve which is defined to contain $g : V \to U$
whenever, for all $f : W \to V$ in $g^*(R)$, the part $S(gf)$ is the
maximal sieve.

```agda
  module _ {U : ⌞ C ⌟} (R : J ʻ U) (S : Patch (ΩJ .fst) ⟦ R ⟧) where
    S' : Sieve C U
    S' .arrows {V} g = elΩ $
      ∀ {W} (f : Hom W V) (hf : f ∈ pullback g ⟦ R ⟧) →
      ∀ {V'} (i : Hom V' W) → i ∈ S .part (g ∘ f) hf .fst

    S' .closed = elim! λ α h → inc λ {W} g hf →
      subst (λ e → ∀ (h : e ∈ R) {V'} (i : Hom V' W) → i ∈ S .part e h .fst)
        (assoc _ _ _) (α (h ∘ g)) hf
```

<!--
```agda
    module _ {V W W'} (f : Hom V U) (hf : f ∈ ⟦ R ⟧) (g : Hom W V) (hfg : f ∘ g ∈ ⟦ R ⟧) {h : Hom W' W} where
      lemma : h ∈ S .part (f ∘ g) hfg .fst ≡ (g ∘ h) ∈ S .part f hf .fst
      lemma = sym (ap (λ e → ⌞ e .fst .arrows h ⌟) (S .patch f hf g hfg))

      module lemma = Equiv (path→equiv lemma)
```
-->

The first thing we have to show is that this pulls back to $S$. This is,
as usual, a proof of biimplication, though in this case both directions
are painful --- and entirely mechanical --- calculations.

```agda
    restrict : ∀ {V} (f : Hom V U) (hf : f ∈ R) → pullback f S' ≡ S .part f hf .fst
    restrict f hf = ext λ {V} h → biimp
      (rec! λ α →
        let
          step₁ : id ∈ S .part (f ∘ h ∘ id) (⟦ R ⟧ .closed hf (h ∘ id)) .fst
          step₁ = subst₂ (λ e e' → id ∈ S .part e e' .fst) (pullr refl) (to-pathp⁻ refl)
            (α id _ id)

          step₂ : ((h ∘ id) ∘ id) ∈ S .part f hf .fst
          step₂ = lemma.to f hf (h ∘ id) (⟦ R ⟧ .closed hf (h ∘ id)) {id} step₁

        in subst (λ e → ⌞ S .part f hf .fst .arrows e ⌟) (cancelr (idr _)) step₂)
      (λ hh → inc λ {W} g hg {V'} i → S .part ((f ∘ h) ∘ g) hg .snd i (max
        let
          s1 : i ∈ S .part (f ∘ h ∘ g) _ .fst
          s1 = lemma.from f hf (h ∘ g) _
            (subst (_∈ S .part f hf .fst) (assoc _ _ _)
              (S .part f hf .fst .closed hh (g ∘ i)))

          q : PathP (λ i → assoc f h g i ∈ R) _ hg
          q = to-pathp⁻ refl
        in transport (λ j → ⌞ S .part (assoc f h g j) (q j) .fst .arrows (idr i (~ j)) ⌟) s1))
```

Finally, we can use this to show that $S'$ is closed.

```agda
    abstract
      S'-closed : is-closed S'
      S'-closed {V} h hb = inc λ {W} f hf {V'} i → S .part (h ∘ f) hf .snd i $
        let
          p =
            pullback (f ∘ i) (pullback h S')     ≡˘⟨ pullback-∘ ⟩
            pullback (h ∘ f ∘ i) S'              ≡⟨ restrict (h ∘ f ∘ i) (subst (_∈ R) (sym (assoc h f i)) (⟦ R ⟧ .closed hf i)) ⟩
            S .part (h ∘ f ∘ i) _ .fst           ≡⟨ ap₂ (λ e e' → S .part e e' .fst) (assoc h f i) (to-pathp⁻ refl) ⟩
            S .part ((h ∘ f) ∘ i) _ .fst         ≡˘⟨ ap fst (S .patch (h ∘ f) hf i (⟦ R ⟧ .closed hf i)) ⟩
            pullback i (S .part (h ∘ f) hf .fst) ∎
        in subst (J ∋_) p (pull (f ∘ i) hb)
```

<!--
```agda
    mk : Section (ΩJ .fst) S
    mk .whole = S' , S'-closed
    mk .glues f hf = Σ-prop-path! (restrict f hf)
```
-->
