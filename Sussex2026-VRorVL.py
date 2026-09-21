import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def imports():
    import marimo as mo
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    return mo, mpl, np, plt


@app.cell(hide_code=True)
def constants():
    # CONSTANTS
    c_maxyears = 20  # Only the most recent 20 years count
    c_maxCR = 22_530  # Maximum SRP is £22,530
    c_minyears = 2  # Minimum years of service for SRP
    c_maxweekly = 751  # Maximum weekly pay considered for SRP is £751
    c_maxVR = 95_000  # Maximum VR is £95,000
    c_multVR = 1.5  # Multiplier for VR
    c_multVL_a = 2.1  # option a) 2.1 weeks pay times years of service
    c_multVL_b = 15.6  # option b) 3.6 months pay = 15.6 weeks pay
    return (
        c_maxCR,
        c_maxVR,
        c_maxweekly,
        c_maxyears,
        c_minyears,
        c_multVL_a,
        c_multVL_b,
        c_multVR,
    )


@app.cell(hide_code=True)
def helper_funcs(c_maxyears, np):
    # HELPER FUNCS

    # ======================================================================================
    # Multiplier as a function of age range
    # ======================================================================================
    def multiplier(agex):
        m = 0.5 if agex < 22 else 1 if agex < 41 else 1.5
        return m

    # ======================================================================================
    # Convert monthly salary to weekly salary
    # ======================================================================================
    def monthly_to_weekly(monthly):
        return monthly * 12 / 52

    # ======================================================================================
    # return 1 or identity matrix depending on type of arg
    # ======================================================================================
    def identity(arg):
        scalarmode = isinstance(arg, float) or isinstance(arg, int)
        I = 1 if scalarmode else np.ones(shape=arg.shape)
        return I

    # ===========================================================
    # Entitled weeks is a function of age and years of service
    # ===========================================================
    def entitled_weeks(age, nyears):
        w = 0
        for y in range(0, nyears):
            if y >= c_maxyears:
                break
            a = age - y
            m = multiplier(
                a
            )  # 0.5, 1, 1.5 depending on age in year of service
            w += m
        return w

    # ===========================================================
    # Net salary from Basic salary
    # ===========================================================
    def net_from_basic(
        basic_annual,
        pension_rate=6.1,
        pension_extra_annual=0,
        student_loan_plan=1,
    ):
        pension_deducted = (
            pension_rate / 100 * basic_annual + pension_extra_annual
        )
        gross_annual = basic_annual - pension_deducted
        personal_allowance = 12570
        tax_brackets = [(50270, 0.20), (float("inf"), 0.40)]
        ni_brackets = [(50270, 0.08), (float("inf"), 0.02)]
        taxable = max(0.0, basic_annual - personal_allowance)

        # TAX
        total_tax = 0.0
        previous_limit = 0.0
        for limit, rate in tax_brackets:
            if taxable > previous_limit:
                taxable_in_bracket = min(taxable, limit) - previous_limit
                total_tax += taxable_in_bracket * rate
                previous_limit = limit
            else:
                break
        # NIC
        total_ni = 0
        previous_limit = 0.0
        for limit, rate in ni_brackets:
            if taxable > previous_limit:
                taxable_in_bracket = min(taxable, limit) - previous_limit
                total_ni += taxable_in_bracket * rate
                previous_limit = limit
            else:
                break

        # STUDENT LOAN
        thresh = (
            2_241
            if student_loan_plan == 1
            else 2_448
            if student_loan_plan == 2
            else basic_annual
        )
        loan_payment = 0.09 * (basic_annual - 12 * thresh)

        return (
            round(gross_annual - total_tax - total_ni - loan_payment, 2),
            total_tax,
            total_ni,
        )

    return entitled_weeks, identity, monthly_to_weekly, net_from_basic


@app.cell(hide_code=True)
def calculators(
    c_maxCR,
    c_maxVR,
    c_maxweekly,
    c_minyears,
    c_multVL_a,
    c_multVL_b,
    c_multVR,
    entitled_weeks,
    identity,
    monthly_to_weekly,
    net_from_basic,
    np,
):
    # ===========================================================
    # CALCULATORS FOR CR, VR, VL
    # Functions can take scalar or vector arguments
    # ===========================================================

    # ===========================================================
    # Compulsory Redundancy = Statutory Redundancy Payment
    # ===========================================================
    def CR(weekly, nweeks, nyears):
        I = identity(weekly)
        # cap the weekly pay at c_maxweekly
        weekly = np.minimum(weekly, c_maxweekly * I)
        # zero payment if min years requirement not met
        if np.isscalar(nyears):
            if nyears < c_minyears:
                return 0
        else:
            weekly[nyears < c_minyears] = 0
        return np.minimum(nweeks * weekly, c_maxCR * I)

    # ===========================================================
    # Voluntary redundancy payment is 1.5 * weekly_pay * entitled_weeks capped at 95k
    # ===========================================================
    def VR(weekly, nweeks):
        I = identity(weekly)
        return np.minimum(c_multVR * nweeks * weekly, c_maxVR * I)

    # ===========================================================
    # Voluntary Leavers payment is the larger of
    # a) 2.1 weeks pay * years of service
    # b) 3.6 months pay = 15.6 weeks pay
    # capped at 95k
    # Note that this is independent of age.
    # ===========================================================
    def VL(weekly, nyears):
        I = identity(weekly)
        VL_a = c_multVL_a * weekly * nyears
        VL_b = c_multVL_b * weekly
        return np.minimum(np.maximum(VL_a, VL_b), c_maxVR * I)

    # ===========================================================
    # 2.5 months additional pay for VR versus VL based on leaving date for VR being later than for VL
    # ===========================================================
    def extra_pay_from_vr(
        monthly, pension_rate=6.1, pension_extra_monthly=0, student_loan_plan=1
    ):
        net_annual = net_from_basic(
            monthly * 12,
            pension_rate=pension_rate,
            pension_extra_annual=12 * pension_extra_monthly,
            student_loan_plan=student_loan_plan,
        )[0]
        return 2.5 * net_annual / 12

    # ===========================================================
    # 2.5 months additional pay for VR versus VL based on leaving date for VR being later than for VL
    # ===========================================================
    def extra_pay_from_vr_net(monthly_net):
        return 2.5 * monthly_net

    # ===========================================================
    # Calculate the CR, VR, and VL26 payments
    # ===========================================================
    def calculate(nyears, age, monthly):

        weekly = monthly_to_weekly(monthly)
        nweeks = entitled_weeks(age, nyears)

        calc_cr = CR(weekly, nweeks, nyears)
        calc_vr = VR(weekly, nweeks)
        calc_vl = VL(weekly, nyears)

        return calc_cr, calc_vr, calc_vl

    return CR, VL, VR, extra_pay_from_vr, extra_pay_from_vr_net


@app.cell(hide_code=True)
def make_plots(
    CR,
    VL,
    VR,
    entitled_weeks,
    extra_pay_from_vr,
    monthly_to_weekly,
    mpl,
    np,
    plt,
):
    # ====================================================
    # Make plots
    # ====================================================
    def plots(
        basic_monthly=0,
        net_monthly=0,
        show_user=0,
        user_age=0,
        user_years=0,
    ):

        test_salary = round(basic_monthly, 2)

        # 1D array: ages
        ages = np.arange(18, 76, 1)

        # 1D array: years of service
        yoss = np.arange(0, 40, 1)

        # calculate weekly pay
        weekly = monthly_to_weekly(test_salary)
        monthly = test_salary

        # ====================================================
        # 2D arrays:
        # W: entitled weeks in 2D array shape
        # P: weekly pay in 2D array shape
        # impmask: impossible values (yos cannot exceed age-18) filled with 0, possible values filled with 1
        # ====================================================
        W = []
        P = []
        impmask = []

        for age in ages:
            wrow = []
            prow = []
            improw = []
            for yos in yoss:
                if age - yos < 18:
                    wy = 1
                    py = 1
                    iy = 0
                else:
                    wy = entitled_weeks(age, yos)
                    py = weekly
                    iy = 1

                wrow.append(wy)
                prow.append(py)
                improw.append(iy)

            W.append(wrow)
            P.append(prow)
            impmask.append(improw)

        # convert 2D lists to numpy arrays

        W = np.array(W)
        P = np.array(P)
        impmask = np.array(impmask)

        # ====================================================
        # 2D arrays of X = ages, Y = years of service
        # ====================================================
        # 2D arrays (NxN squares, each row a copy of the 1D arrays we defined above)
        X, Y = np.meshgrid(ages, yoss)

        # ====================================================
        # Calculate payments for the three different scenarios
        # ====================================================

        # Y.T : Pass the Transpose ofthe "years of service" array (rows <=> columns)

        # Compulsory (statutory redundancy pay)
        calc_cr = CR(P, W, Y.T)
        # in terms of month of salary
        calc_cr_sal = calc_cr / (
            test_salary / 12 * np.ones(shape=calc_cr.shape)
        )

        # Voluntary redundancy payment
        calc_vr = VR(P, W)
        # in terms of month of salary
        calc_vr_sal = calc_vr / (
            test_salary / 12 * np.ones(shape=calc_vr.shape)
        )

        # Voluntary redundancy payment plus 2.5 months pay (different timings for VR and VL)
        calc_vr_extra = calc_vr + extra_pay_from_vr(net_monthly)
        calc_vr_extra_sal = calc_vr_extra / (
            test_salary / 12 * np.ones(shape=calc_vr_extra.shape)
        )

        # Voluntary Leavers 26 payment
        calc_vl = VL(P, Y.T)
        # in terms of month of salary
        calc_vl_sal = calc_vl / (
            test_salary / 12 * np.ones(shape=calc_vl.shape)
        )

        # ====================================================
        # Make plots
        # ====================================================

        # set up figure: 4 sublots
        fig, axs = plt.subplots(
            ncols=3,
            nrows=2,
            figsize=(18, 12),
            layout="tight",
            sharex=False,
            sharey=False,
        )

        # figure name for this test_salary
        figname = f"VLminusVR_{test_salary}k_extra.png"

        # Styles and labels for the four different subplots
        scenarios = ["CR", "VR", "VL", "VL-VR", "VL-VR+"]
        calcs = [
            calc_cr,
            calc_vr,
            calc_vl,
            calc_vl - calc_vr,
            calc_vl - calc_vr_extra,
        ]

        tit_cr = r"$\bf{CR\; Payment\; (£k)} $"
        tit_vr = r"$\bf{VR\; Payment\; (£k)} $"
        tit_vl = r"$\bf{VL26\; Payment\; (£k)} $"
        tit_diff = r"$\bf{Difference\; VL26\,-\,VR\; (normalised)}$"
        tit_diff2 = r"$\bf{Difference\; VL26\,-\,VR^{+}\; (normalised)}$"
        titles = [tit_cr, tit_vr, tit_vl, tit_diff, tit_diff2]

        cmaps = ["Reds", "Oranges", "Blues", "bwr_r", "bwr_r"]

        tcols = ["#7F1E1A", "#DD6D30", "#689ECA", "k", "k"]

        # Loop over the different scenarios
        for i, scen in enumerate(scenarios):
            # flatten the axes array as it is 2x2 for 2 rows and 2 columns
            ax = axs.ravel()[i]

            # get ten values of the color map for this scenario
            cmap = plt.get_cmap(cmaps[i], 10)

            # Set Z to the 2D array of calculated payments for this scenario
            Z = calcs[i]

            # The VL-VR plot is normalised
            if scen == "VL-VR" or scen == "VL-VR+":
                # change negative values to -1, positive to +1, leave zero as is.
                Z = np.sign(Z)
                # multiply by the impossibility mask to get 0 for impossible values
                Z = Z * impmask
                # orange, white, blue
                levels = [-1.1, -0.03, 0.03, 1.1]
                colors = ["#f87d2a", "w", "#59a2cf"]
                # Plot
                cs = ax.contourf(ages, yoss, Z.T, levels=levels, colors=colors)
                # colorbar
                cbar = plt.colorbar(
                    cs, ax=ax, location="top", pad=0.01, spacing="proportional"
                )
                if scen == "VL-VR":
                    cbar.ax.set_xticks(
                        [-0.7, 0, 0.7],
                        labels=["VR pays more", "", "VL26 pays more"],
                    )  # , fontsize=24)
                else:
                    cbar.ax.set_xticks(
                        [-0.7, 0, 0.7],
                        labels=[
                            r"$VR^{\bf{+}}$" + " pays more",
                            "",
                            "VL26 pays more",
                        ],
                    )  # , fontsize=24)
                cbar.ax.set_xlabel(titles[i], fontsize=18, labelpad=12)

                cc = ax.contour(
                    ages,
                    yoss,
                    Z.T,
                    levels=levels,
                    colors="black",
                    linewidths=2,
                )

            # the other plots are straight up
            else:
                Z = Z * impmask
                levels = np.linspace(1e-3, Z.max(), cmap.N + 1)
                levels = np.insert(levels, 0, 0.0, axis=0)
                colors = ["#ffffff"]
                for j in range(cmap.N):
                    c = mpl.colors.rgb2hex(cmap(j))
                    colors.append(c)

                cs = ax.contourf(ages, yoss, Z.T, levels=levels, colors=colors)
                cbar = plt.colorbar(cs, ax=ax, location="top", pad=0.01)
                cbar.ax.set_xlabel(
                    titles[i], fontsize=18, labelpad=12
                )  # , color=tcols[i])

                cc = ax.contour(
                    ages,
                    yoss,
                    Z.T,
                    levels=levels,
                    colors="black",
                    linewidths=0.5,
                )

            if show_user:
                ax.plot(
                    user_age,
                    user_years,
                    marker="*",
                    markersize=15,
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markeredgewidth=1,
                )

            ax.set_xlabel(r"Age ", fontsize=18)
            ax.set_ylabel(r"Years of service", fontsize=18)

        axn = axs.ravel()[5]
        axn.axis("off")

        axn.text(
            0,
            0.8,
            "CR: Statutory Redundancy Payment ",
            fontsize=18,
            weight="bold",
            color=tcols[0],
        )
        axn.text(
            0.1,
            0.73,
            "f(age, years of service, salary)",
            fontsize=16,
            style="italic",
        )
        axn.text(
            0,
            0.6,
            "VR: Voluntary Redundancy Payment ",
            fontsize=18,
            weight="bold",
            color=tcols[1],
        )
        axn.text(
            0.1,
            0.53,
            " f(age, years of service, salary)",
            fontsize=16,
            style="italic",
        )
        axn.text(
            0,
            0.4,
            "VL26: Voluntary Leavers 2026 Payment ",
            fontsize=18,
            weight="bold",
            color=tcols[2],
        )
        axn.text(
            0.1,
            0.33,
            " f(years of service, salary)",
            fontsize=16,
            style="italic",
        )
        axn.text(
            0,
            0.2,
            r"VR$^{+}$" + ": VR+2.5 months net pay",
            fontsize=18,
            weight="bold",
            color=tcols[4],
        )
        axn.text(
            0.1,
            0.13,
            "  (for difference in leaving dates VR-VL)",
            fontsize=16,
            style="italic",
        )
        fig.suptitle(
            f"Example for monthly basic salary = £{test_salary:,.0f}",
            fontsize=24,
        )
        return fig

    return (plots,)


@app.cell
def _(mo):
    salary_option = mo.ui.number(
        start=1_000,
        stop=125_000,
        step=1,
        value=3_333,
    )
    salary_ma_option = mo.ui.dropdown(
        options=["Monthly", "Annual"],
        value="Monthly",
    )
    return salary_ma_option, salary_option


@app.cell
def inputs(mo, salary_ma_option, salary_option):
    net_salary_option = mo.ui.number(
        start=1_000,
        stop=125_000,
        step=1,
        value=salary_option.value * 0.74,
    )
    net_salary_ma_option = mo.ui.dropdown(
        options=["Monthly", "Annual"],
        value=salary_ma_option.value,
    )

    age_check = mo.ui.number(
        start=18,
        stop=80,
        step=1,
        value=35,
    )
    years_check = mo.ui.number(
        start=1,
        stop=40,
        step=1,
        value=7,
    )

    show_on_plot = mo.ui.radio(
        options=["Yes", "No"],
        value="Yes",
        label="Would you like to show your point on the plots?",
        inline=True,
    )
    return (
        age_check,
        net_salary_ma_option,
        net_salary_option,
        show_on_plot,
        years_check,
    )


@app.cell
def global_1(
    age_check,
    entitled_weeks,
    monthly_to_weekly,
    net_salary_ma_option,
    net_salary_option,
    salary_ma_option,
    salary_option,
    years_check,
):
    # global

    basic_is_monthly = 1 if salary_ma_option.value == "Monthly" else 0

    net_is_monthly = 1 if net_salary_ma_option.value == "Monthly" else 0

    if basic_is_monthly:
        monthly = salary_option.value
    else:
        monthly = salary_option.value / 12

    weekly = monthly_to_weekly(monthly)

    nweeks = entitled_weeks(age_check.value, years_check.value)

    if net_is_monthly:
        nmonthly = net_salary_option.value
    else:
        nmonthly = net_salary_option.value / 12
    return monthly, nmonthly, nweeks, weekly


@app.cell(hide_code=True)
def refs(mo):
    def refs_stack():

        ref_uni_vr_calculator = mo.md(
            """
        **VR Calculator**: The University Voluntary Redundancy calculator is on [**SharePoint**](https://universityofsussex.sharepoint.com/sites/ChangeAndTransformation). Select button **Voluntary Redundancy (VR) Calculator**'.
        """
        )

        ref_gov_cr_calculator = mo.md(
            """
            **CR Calculator**: The UK Government's Statutory Redundancy Payment Calculator is on [**gov.uk**](https://www.gov.uk/calculate-your-redundancy-pay).  *Compulsory Redundancy (CR) and Statutory Redundancy are the same thing.*
            """
        )

        ref_myview = mo.md(
            """
            **VL26 Calculator** The University VL26 Calculator is available from login to [**MyView**](https://myviewhr.sussex.ac.uk/dashboardlive/dashboard-ui/index.html). **Voluntary Leavers-> Voluntary Leavers 2026 -> Payment Calculator**.
            """
        )

        ref_uni_notes = mo.md(
            """
            **Information on University CR and VR terms** is on [**SharePoint**](https://universityofsussex.sharepoint.com/sites/ChangeAndTransformation): select button **Voluntary and Compulsory Redundancy Terms**
            """
        )

        ref_timeline = mo.md(
            """
            **The timeline for VL26 and VR** is on [**SharePoint**](https://universityofsussex.sharepoint.com/sites/ChangeAndTransformation): select button **Timeline -> Update as of 16 July 2026 -> Timelines --- additional detail**.
            """
        )

        references = mo.vstack(
            [
                ref_uni_vr_calculator,
                ref_gov_cr_calculator,
                ref_myview,
                ref_uni_notes,
                ref_timeline,
            ]
        )

        return references

    return (refs_stack,)


@app.cell(hide_code=True)
def disclaimer(mo):
    def disclaimer():
        bug = mo.icon("lucide:bug", size=15, color="red")

        disc = mo.md(
            f"I made this in my free time because I was confused. Please email me if you find a {bug}. References/links at bottom. Lily: l.asquith@sussex.ac.uk."
        )

        return disc

    return (disclaimer,)


@app.cell
def monthly_weekly_annual():
    return


@app.cell(hide_code=True)
def inputs_explainers(
    age_check,
    mo,
    net_salary_ma_option,
    net_salary_option,
    salary_ma_option,
    salary_option,
    years_check,
):
    def input_explainer_stack():

        ksalary_option = mo.md(f"Basic Pay (£): ")
        kage_check = mo.md(f"Age: ")
        kyears_check = mo.md(f"Years of service: ")
        knet_salary_option = mo.md(f"Net Pay (£): ")

        kinputs = mo.vstack(
            [
                ksalary_option,
                kage_check,
                kyears_check,
                knet_salary_option,
            ],
            justify="start",
        )

        bs = mo.hstack(
            [
                salary_option,
                salary_ma_option,
            ],
            justify="start",
        )

        ns = mo.hstack(
            [
                net_salary_option,
                net_salary_ma_option,
            ],
            justify="start",
        )

        inputs = mo.vstack(
            [
                bs,
                age_check,
                years_check,
                ns,
            ],
            justify="start",
        )

        explain_salary_option = mo.md(
            r"""$\longleftarrow\;$ [**MyView**](https://myviewhr.sussex.ac.uk/dashboardlive/dashboard-ui/index.html) payslip: **'Basic Pay - Spinal Salary'** is your Monthly value. *Used in all calculations.*"""
        )

        explain_age_check = mo.md(
            r"""$\longleftarrow$ Age in years **on 1 July 2026**. *Used in the calculation of CR and VR.*"""
        )

        explain_years_check = mo.md(
            r"""$\longleftarrow$ **Full** years of service **on 1 July 2026**. *Used in all calculations.*"""
        )

        explain_net_salary_option = mo.md(
            r"""$\longleftarrow$ [**MyView**](https://myviewhr.sussex.ac.uk/dashboardlive/dashboard-ui/index.html) payslip: **'Net Pay'** is your Monthly value. *Used to calculate VR+ (VR plus 2.5 months net pay).*"""
        )

        explainers = mo.vstack(
            [
                explain_salary_option,
                explain_age_check,
                explain_years_check,
                explain_net_salary_option,
            ],
            justify="start",
        )

        input_explainer = mo.hstack(
            [
                kinputs,
                inputs,
                explainers,
            ],
            justify="start",
            widths=[1, 1, 3],
        )

        return input_explainer

    return (input_explainer_stack,)


@app.cell(hide_code=True)
def _(age_check, mo, monthly, nmonthly, plots, show_on_plot, years_check):
    show_user = 1 if show_on_plot.value == "Yes" else 0

    plot_output = mo.as_html(
        plots(
            monthly,
            nmonthly,
            show_user,
            age_check.value,
            years_check.value,
        )
    )
    return (plot_output,)


@app.cell(hide_code=True)
def _(
    CR,
    VL,
    VR,
    extra_pay_from_vr_net,
    mo,
    nmonthly,
    nweeks,
    weekly,
    years_check,
):
    def calcs_stack():

        your_cr = CR(weekly, nweeks, years_check.value)

        your_vr = VR(weekly, nweeks)

        your_crx = your_cr + extra_pay_from_vr_net(nmonthly)

        your_vrx = your_vr + extra_pay_from_vr_net(nmonthly)

        your_vl = VL(weekly, years_check.value)

        #  tcols = ["#7F1E1A", "#DD6D30", "#689ECA", "k", "k"]
        calcs_output = mo.vstack(
            [
                mo.md(r"""$\;\;$""" + f"CR=£{your_cr:,.0f}, ").style(
                    {"color": "#8f211d", "font-weight": "bold"}
                ),
                mo.md(r"""$\;\;$""" + f"VR=£{your_vr:,.0f}, ").style(
                    {"color": "#DD6D30", "font-weight": "bold"}
                ),
                mo.md(r"""$\;\;$""" + f"VL26=£{your_vl:,.0f},").style(
                    {"color": "#689ECA", "font-weight": "bold"}
                ),
                mo.md(r"""$\;\;$""" + f"VR+ =£{your_vrx:,.0f} ").style(
                    {"color": "black", "font-weight": "bold"}
                ),
            ],
            justify="start",
        )
        return calcs_output

    return (calcs_stack,)


@app.cell(hide_code=True)
def slide(
    calcs_stack,
    disclaimer,
    input_explainer_stack,
    mo,
    plot_output,
    refs_stack,
    show_on_plot,
):
    # Define the slide as a vertical stack of objects with horizontal lines between them
    def slide():

        full_stack = mo.vstack(
            [
                mo.md(f"### **This is not an official tool**"),
                disclaimer(),
                mo.md("---"),
                mo.md(f"### **Pay, Age, and Years of Service** "),
                input_explainer_stack(),
                mo.md("---"),
                mo.md(f"### **Payment Estimates** "),
                calcs_stack(),
                mo.md("---"),
                mo.md(f"### **Plots** "),
                show_on_plot,
                plot_output,
                mo.md("---"),
                mo.md(f"### **References** "),
                refs_stack(),
            ],
            justify="start",
        )

        return full_stack

    return (slide,)


@app.cell
def _(slide):
    slide()
    return


if __name__ == "__main__":
    app.run()