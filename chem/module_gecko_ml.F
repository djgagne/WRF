module module_gecko_ml
    use module_neural_net
    implicit none
    
    type neural_net_gecko
        type(Dense), allocatable :: gecko_nn(:)
        real(8), allocatable :: input_scale_values(:, :)
        real(8), allocatable :: output_scale_values(:, :)
        integer :: in_size
        integer :: out_size
        integer :: batch_size
    end type neural_net_gecko

    type(neural_net_gecko), save :: nn_gecko

    ! init gecko ml will need to be called from chemics_init.F or whatever init
    ! module is used before WRF Chem starts forward integration
    subroutine init_gecko_ml(neural_net_path)
        character(len=*), intent(in) :: neural_net_path
        neural_net_gecko%batch_size = 1
        call load_all_scale_values(neural_net_path // "gecko_input_scale_values.csv", &
             nn_gecko%input_scale_values)
        call load_all_scale_values(neural_net_path // "gecko_output_scale_values.csv", &
             nn_gecko%output_scale_values)
        call init_neural_net(neural_net_path // "gecko_neural_net.nc", &
             batch_size, nn_gecko%gecko_nn)
        nn_gecko%in_size = size(nn_gecko%input_scale_values, 1)
        nn_gecko%out_size = size(nn_gecko%output_scale_values, 1)
    end subroutine init_gecko_ml

    ! run_gecko_neural_net needs to be called by one of the driver files. I need
    ! to know what the appropriate WRF variables are for each variable and their 
    ! shapes.
    subroutine run_gecko_neural_net(precursor, aerosol, gas, temperature, &
        zenith, pre_exist_aerosols, o3, nox, oh, ims, ime, jms, jme, kms, kme, &
        its, ite, jts, jte, kts, kte, &
        bins)
        integer intent(in) :: ims, ime, jms, jme, kms, kme
        ! these are my best guesses at the shape of these fields, can adjust
        real, dimension(ims:ime, kms:kme, jms:jme), intent(inout) :: precursor
        real, dimension(ims:ime, kms:kme, jms:jme,  bins), intent(inout) :: aerosol, gas
        real, dimension(ims:ime, kms:kme, jms:jme), intent(in) :: temperature, &
            zenith, pre_exist_aerosols, o3, nox, oh
        ! zenith may be a 2D field, need to check
        do i=its:ite
            do k=kts:kte
                do j=jts:jte
                    call run_gecko_neural_net_1d(precursor(i, k, j), &
                                                 gas(i, k, j), &
                                                 aerosol(i, k, j), &
                                                 temperature(i, k, j), &
                                                 zenith(i, k, j), &
                                                 pre_exist_aerosols(i, k, j), &
                                                 o3(i, k, j), &
                                                 nox(i, k, j), &
                                                 oh(i, k, j), &
                                                 bins)
                end do
            end do
        end do
    end subroutine run_gecko_neural_net

    subroutine run_gecko_neural_net_1d(precursor, gas, aerosol, temperature, &
        zenith, pre_exist_aerosols, o3, nox, oh, bins)
        integer, intent(in) :: bins, i
        real, intent(inout) :: precursor
        real, dimension(bins), intent(inout) :: gas, aerosol
        real, intent(in) :: temperature, zenith, pre_exist_aerosols, o3, nox, oh
        real, dimension(batch_size, gecko_nn%in_size) :: nn_input, nn_scaled_input
        real, dimension(batch_size, gecko_nn%output_size) :: nn_output, nn_scaled_output
        ! fill ANN input array
        ! we may need to modify the code to take the log of the precursor here
        nn_input(1, 1) = precursor
        i = 2
        ! I tried to enable support for varying numbers of gas and aerosol bins
        nn_input(1, i: i + size(gas)) = gas
        i = i + size(gas)
        nn_input(1, i: i + size(aerosol)) = aerosol
        i = i + size(aerosol)
        nn_input(1, i: i + 6) = (/ temperature, zenith, pre_exist_aerosols, &
            o3, nox, oh /)
        ! scale input values
        call standard_scaler_transform(nn_input, nn_gecko%input_scale_values, &
                                       nn_scaled_input)
        ! call the neural network perdiction function
        call neural_net_predict(nn_scaled_input, nn_gecko%gecko_nn, &
            nn_scaled_output)
        ! inverse scale the output values
        call standard_scaler_inverse_transform(nn_scaled_output, &
            nn_gecko%output_scale_values, nn_output)
        i = 1
        ! may need to un-log the precursor
        precursor = nn_output(1, i)
        i = 2
        gas = nn_output(1, i: i + size(gas))
        i = i + size(gas)
        aerosol = nn_output(1, i: i + size(aerosol)
    end subroutine run_gecko_neural_net_1d

end module module_gecko_ml

