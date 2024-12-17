<!--
```agda
open import Cat.Instances.Shape.Terminal
open import Cat.Instances.Coalgebras
open import Cat.Diagram.Limit.Base
open import Cat.Diagram.Limit.Cone
open import Cat.Functor.Kan.Unique
open import Cat.Functor.Naturality
open import Cat.Functor.Coherence
open import Cat.Functor.Constant
open import Cat.Functor.Kan.Base
open import Cat.Diagram.Comonad
open import Cat.Displayed.Total
open import Cat.Functor.Compose
open import Cat.Prelude

import Cat.Functor.Reasoning as Func
import Cat.Reasoning
```
-->

```agda
module Cat.Instances.Coalgebras.Limits
  {o ℓ} (C : Precategory o ℓ) {F : Functor C C} (W : Comonad-on F)
  where
```

<!--
```agda
open Cat.Reasoning C

open Total-hom
open _=>_

open Coalgebra-on
open Comonad-on W using (module comult ; module counit ; W-∘ ; W-id ; W₀ ; W₁)

private module W = Func F
```
-->

# Arbitrary limits of coalgebras

This module concerns itself with the more general construction of
[[limits]] in a [[category of coalgebras]], as mentioned in the (more
focused) construction of [[finite limits of coalgebras]]. Namely, if
$(W, \eps, \delta)$ is a [[comonad]] on $\cC$ which, as a functor,
preserves $\cI$-shaped limits, then the forgetful functor $U : \cC_W \to
\cC$ preserves _and_ reflects those same limits.

<!--
```agda
module
  _ {oi ℓi} {I : Precategory oi ℓi}
    (pres : ∀ (G : Functor I C) {X η} (l : is-ran !F G X η) → preserves-ran F l)
  where
```
-->

We start by showing reflection: Let $F : \cI \to \cC_W$ be a diagram of
$W$-coalgebras, $L$ be a coalgebra, and $\phi$ be a natural
transformation $\Delta(L) \to F$. If $(UL, U\phi)$ is the limit of $UF$
in $\cC$, then $(L, \phi)$ is the limit of $F$ in $\cC_W$.

```agda
  is-limit-coalgebra
    : ∀ (F : Functor I (Coalgebras W)) {K phi}
    → (l : is-ran !F (πᶠ _ F∘ F) (πᶠ _ F∘ K) (nat-assoc-from (πᶠ _ ▸ phi)))
    → reflects-ran (πᶠ (Coalgebras-over W)) l
  is-limit-coalgebra F {K} {phi} l = to-is-limitp mk fixup where
```

It will suffice to show the following: if $\eta_j : x \to Fj$ is a
natural family of coalgebra homomorphisms, then one may construct a map
$\nu : x \to L$ satisfying $\phi_j \nu = \eta_j$. For if we are given
such a map, we can show that $\nu$ is a coalgebra homomorphism $x \to
L$. Since $WL$ is a limit, we may argue for preservation componentwise
against the projections $W\phi_j$, and we have

$$
\begin{align*}
& \phantom{=} W\phi_j \circ W\nu \circ x  \\
&          =  Fj \circ \eta_j             \\
&          =  W\phi_j \circ L \circ \nu
\end{align*}
$$

so that $W\nu \circ x = L \circ \nu$, as required. Showing that $\nu$ is
indeed the unique map factoring $\eta_j$ through $L$ is done through
exactly the same style of argument.

<!--
```agda
    module K = Functor K
    module F = Functor F

    module l = is-limit l
    module l' = is-limit (pres _ l)
    open make-is-limit

    module
      _ {x : Coalgebras.Ob W}
        (eta : (j : Precategory.Ob I) → Coalgebras.Hom W x (F.₀ j))
        (nat : ∀ {x y} (f : Precategory.Hom I x y) → Coalgebras._∘_ W (F.₁ f) (eta x) ≡ eta y)
      where
```
-->

That's very well and good, but does such $\nu$ exist? Well, yes: much
like in the finite case, we note that $Fj \circ \eta_j$ is also natural
in $j$, so that the limit factorisation

$$
\langle Fj \circ \eta_j \rangle_j : x \to WL
$$

exists. By composing with $\eps$, we obtain a map $x \to WL \to L$, and
with a short calculation we establish that this map satisfies the
commutativity condition we remarked was sufficient above; we're done!

```agda
      opaque
        ν : Hom (x .fst) (K.₀ tt .fst)
        ν = counit.ε _ ∘ l'.universal (λ j → F.₀ j .snd .ρ ∘ eta j .hom) λ {x} {y} f →
          W₁ (F.₁ f .hom) ∘ F.₀ x .snd .ρ ∘ eta x .hom ≡⟨ pulll (F.₁ f .preserves) ⟩
          (F.₀ y .snd .ρ ∘ F.₁ f .hom) ∘ eta x .hom    ≡⟨ pullr (ap hom (nat _)) ⟩
          F.₀ y .snd .ρ ∘ eta y .hom                   ∎

        ν-β : ∀ {j} → phi .η j .hom ∘ ν ≡ eta j .hom
        ν-β {j} =
          phi .η j .hom ∘ ν                         ≡⟨ pulll (sym (counit.is-natural _ _ _)) ⟩
          (counit.ε _ ∘ l'.ψ j) ∘ l'.universal _ _  ≡⟨ pullr (l'.factors _ _) ⟩
          counit.ε _ ∘ F.₀ j .snd .ρ ∘ eta j .hom   ≡⟨ cancell (F.₀ _ .snd .ρ-counit) ⟩
          eta j .hom                                ∎
```

<!--
```agda
    mk : make-is-limit F (K.₀ tt)
    mk .ψ j .hom       = l.ψ j
    mk .ψ j .preserves = phi .η j .preserves
    mk .commutes f = ext (l.commutes f)
    mk .universal eta nat .hom = ν eta nat
    mk .universal eta nat .preserves = l'.unique₂ _
      (λ f → pulll (F.₁ f .preserves) ∙ pullr (ap hom (nat _)))
      (λ j → W.pulll (ν-β eta nat) ∙ eta j .preserves)
      (λ j → pulll (phi .η j .preserves) ∙ pullr (ν-β eta nat))
    mk .factors eta nat = ext (ν-β eta nat)
    mk .unique eta nat other comm = ext (l.unique₂ _
      (λ f → ap hom (nat f)) (λ j → ap hom (comm j)) (λ j → ν-β eta nat))

    abstract
      fixup : ∀ {j} → mk .ψ j ≡ phi .η j
      fixup = ext refl
```
-->

```agda
  Coalgebra-on-limit
    : (F : Functor I (Coalgebras W))
    → (L : Limit (πᶠ (Coalgebras-over W) F∘ F))
    → Coalgebra-on W (Limit.apex L)
  Coalgebra-on-limit F L = coalg module Coalgebra-on-limit where
```

It remains to show that, if $F$ and $L$ are as before, then the
coalgebra structures on $F(-)$ assemble into a coalgebra structure on
$L$. Fundamentally, this amounts to construct a map $L \to WL$.

<!--
```agda
    private
      module L   = Limit L
      module L'  = is-limit (pres (πᶠ _ F∘ F) L.has-limit)
      module L'' = is-limit (pres _ (pres (πᶠ _ F∘ F) L.has-limit))
      module F   = Functor F
      open counit using (ε)
```
-->

Since $W$ preserves $L$, we may treat maps $X \to WL$ as “tuplings” of
morphisms $X \to Fj$, and test equality componentwise. Tupling the maps
$L \xto{\psi} Fj \to WFj$, where the map $\psi_j$ is the projection from
the limit, we obtain a morphism $\nu : L \to WL$, uniquely characterised
by having $W(\psi_j) \circ \nu = F(j) \circ \psi_j$ for every $j$.

```agda
    opaque
      ν : Hom L.apex (W₀ L.apex)
      ν = L'.universal (λ j → F.₀ j .snd .ρ ∘ L.ψ j) λ {x} {y} h →
        W₁ (F.₁ h .hom) ∘ F.₀ x .snd .ρ ∘ L.ψ x ≡⟨ pulll (F.₁ h .preserves) ⟩
        (F.₀ y .snd .ρ ∘ F.₁ h .hom) ∘ L.ψ x    ≡⟨ pullr (sym (L.eps .is-natural _ _ _) ∙ elimr L.Ext.F-id) ⟩
        F.₀ y .snd .ρ ∘ L.ψ y                   ∎

      ν-β : ∀ {j} → W₁ (L.ψ j) ∘ ν ≡ F.₀ j .snd .ρ ∘ L.ψ j
      ν-β = L'.factors _ _
```

Since we may again reason componentwise to establish compatibility of
$\nu$ with $\eps$ and $\delta$, these will follow from the naturality of
each of $W$'s structural transformations, and from compatibility of each
of the $Fj$'s coalgebra maps. We comment on compatibility with the
counit, and omit the rest of the proof for space.

```agda
    coalg : Coalgebra-on W (Limit.apex L)
    coalg .ρ = ν
    coalg .ρ-counit = L.unique₂ _
      (λ f → sym (L.eps .is-natural _ _ f) ∙ elimr L.Ext.F-id)
```

To show that $\eps \nu = \id$, it will suffice to show that $\psi_j \eps
\nu = \psi_j$. But we have

```agda
      (λ j → L.ψ j ∘ ε _ ∘ ν             ≡⟨ pulll (sym (counit.is-natural _ _ _)) ⟩
             (ε _ ∘ W₁ (L.ψ j)) ∘ ν      ≡⟨ pullr ν-β ⟩
             ε _ ∘ F.₀ j .snd .ρ ∘ L.ψ j ≡⟨ cancell (F.₀ j .snd .ρ-counit) ⟩
             L.ψ j                       ∎)
      (λ j → idr (L.ψ j))
```

<!--
```agda
    coalg .ρ-comult = L''.unique₂ _
      (λ f → W.extendl (F.₁ f .preserves) ∙ ap₂ _∘_ refl
        ( pulll (F.₁ f .preserves)
        ∙ pullr (sym (L.eps .is-natural _ _ f) ∙ elimr L.Ext.F-id)))
      (λ j → W.extendl ν-β ∙ ap₂ _∘_ refl ν-β)
      (λ j → pulll (sym (comult.is-natural _ _ _))
          ·· pullr ν-β
          ·· extendl (sym (F.₀ j .snd .ρ-comult)))

  open Ran
  open is-ran
```
-->

```agda
  Coalgebra-limit
    : (F : Functor I (Coalgebras W))
    → Limit (πᶠ (Coalgebras-over W) F∘ F)
    → Limit F
  Coalgebra-limit F lim .Ext = Const (_ , Coalgebra-on-limit F lim)
```

It remains to show that the projection maps $\psi_j : L \to Fj$ are
coalgebra homomorphisms for the "lifted" structure defined above. But
this condition is precisely $W(\psi_j) \circ \nu = Fj \circ \psi_j$,
i.e., the defining property of $\nu$!

```agda
  Coalgebra-limit F lim .eps .η x .hom       = lim .eps .η x
  Coalgebra-limit F lim .eps .η x .preserves = Coalgebra-on-limit.ν-β F lim
  Coalgebra-limit F lim .eps .is-natural x y f = ext $
    ap₂ _∘_ refl (sym (lim .Ext .Functor.F-id)) ∙ lim .eps .is-natural x y f
```

<!--
```agda
  Coalgebra-limit F lim .has-ran = is-limit-coalgebra F $ natural-isos→is-ran
    idni idni rem₁
    (Nat-path λ x → idl _ ∙ elimr (elimr refl ∙ lim .Ext .Functor.F-id))
    (lim .has-ran)

    where
    open make-natural-iso

    rem₁ : lim .Ext ≅ⁿ πᶠ (Coalgebras-over W) F∘ Const (_ , Coalgebra-on-limit F lim)
    rem₁ = to-natural-iso λ where
      .eta x → id
      .inv x → id
      .eta∘inv x → idl id
      .inv∘eta x → idl id
      .natural x y f → eliml refl ∙ intror (lim .Ext .Functor.F-id)
```
-->
