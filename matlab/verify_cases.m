% Independent MATLAB ode113 implementation; no Python functions are called.
% From the repository root in MATLAB: run('matlab/verify_cases.m')
% Compare solvers by integrating the same fitted parameters.
scriptFolder = fileparts(mfilename('fullpath'));
inputFile = fullfile(scriptFolder,'..','results','matlab_cases.json');
assert(isfile(inputFile),'First run the Python pipeline to export matlab_cases.json.');
payload = jsondecode(fileread(inputFile));
if isstruct(payload) && isfield(payload,'cases'), cases = payload.cases; else, cases = payload; end
tolPairs = [1e-9,1e-11; 1e-12,1e-13];
rows = cell(0,24);
answers = NaN(numel(cases),2,3);
for j = 1:numel(cases)
    c = cases(j);
    for k = 1:size(tolPairs,1)
        rel = tolPairs(k,1); ab = tolPairs(k,2);
        [status,result] = integrateCase(c,rel,ab);
        answers(j,k,:) = result;
        py = [c.python.tg,c.python.yg,c.python.zg];
        delta = abs(result-py);
        analytic = [NaN NaN NaN];
        if c.cd==0 && norm(c.b)==0 && c.v0(1)>0
            tt = (c.target_x-c.r0(1))/c.v0(1);
            analytic = [tt,c.r0(2)+c.v0(2)*tt,c.r0(3)+c.v0(3)*tt-0.5*c.constants.g*tt^2];
        end
        da = abs(result-analytic); dpa = abs(py-analytic);
        pass = strcmp(status,'crossed') && delta(1)<1e-4 && all(delta(2:3)<1e-3);
        rows(end+1,:) = {char(c.name),k,rel,ab,status,py(1),py(2),py(3),result(1),result(2),result(3),delta(1),delta(2),delta(3),analytic(1),analytic(2),analytic(3),da(1),da(2),da(3),dpa(1),dpa(2),dpa(3),pass}; %#ok<SAGROW>
    end
end
names = {'name','tolerance_level','relative_tolerance','absolute_tolerance','status','python_tg_s','python_yg_m','python_zg_m','matlab_tg_s','matlab_yg_m','matlab_zg_m','abs_dt_s','abs_dy_m','abs_dz_m','analytic_tg_s','analytic_yg_m','analytic_zg_m','matlab_analytic_dt_s','matlab_analytic_dy_m','matlab_analytic_dz_m','python_analytic_dt_s','python_analytic_dy_m','python_analytic_dz_m','engineering_pass'};
comparison = cell2table(rows,'VariableNames',names);
writetable(comparison,fullfile(scriptFolder,'matlab_comparison.csv'));
stable = squeeze(abs(answers(:,2,:)-answers(:,1,:)));
if numel(cases)==1, stable=reshape(stable,1,3); end
stability = table(string({cases.name})',stable(:,1),stable(:,2),stable(:,3),'VariableNames',{'name','tightening_dt_s','tightening_dy_m','tightening_dz_m'});
writetable(stability,fullfile(scriptFolder,'matlab_tolerance_stability.csv'));
% Differences between a texture scenario and the main scenario also agree.
mainIdx = find(strcmp(string({cases.name}),'full_MAP_wide'),1);
if ~isempty(mainIdx)
    offsetRows=cell(0,7);
    for j=1:numel(cases)
        pyoff=[cases(j).python.tg-cases(mainIdx).python.tg,cases(j).python.yg-cases(mainIdx).python.yg,cases(j).python.zg-cases(mainIdx).python.zg];
        moff=reshape(answers(j,2,:)-answers(mainIdx,2,:),1,3);
        offsetRows(end+1,:)={char(cases(j).name),pyoff(1),pyoff(2),pyoff(3),moff(1),moff(2),moff(3)}; %#ok<SAGROW>
    end
    writetable(cell2table(offsetRows,'VariableNames',{'name','python_offset_t_s','python_offset_y_m','python_offset_z_m','matlab_offset_t_s','matlab_offset_y_m','matlab_offset_z_m'}),fullfile(scriptFolder,'matlab_offsets.csv'));
end
fid=fopen(fullfile(scriptFolder,'matlab_environment.txt'),'w');
fprintf(fid,'MATLAB %s\nInput: %s\nSolver: ode113\nMaxStep: 0.01 s\nGround: descending z-R event; initial upward launch advanced 1e-8 s before activating events.\n',version,inputFile);
fclose(fid);
disp(comparison); disp(stability);
assert(all(comparison.engineering_pass),'MATLAB/Python engineering tolerance failed.');
assert(all(stable(:,1)<1e-4) && all(all(stable(:,2:3)<1e-3)),'Tolerance tightening stability failed.');

function [status,result] = integrateCase(c,rel,ab)
    q0=[c.r0(:);c.v0(:)]; result=[NaN NaN NaN]; status='not_crossed';
    assert(c.target_x>q0(1),'Target must be ahead of the initial point.');
    if q0(3)<c.constants.R-1e-10 || (q0(3)<=c.constants.R+1e-10 && q0(6)<=0)
        status='grounded'; return
    end
    opts=odeset('RelTol',rel,'AbsTol',ab,'MaxStep',0.01);
    t0=0;
    if q0(3)<=c.constants.R+1e-10
        % MATLAB treats roots at the initial instant specially. Take a tiny
        % upward launch step to distinguish release from descending impact.
        t0=1e-8;
        [~,qq]=ode113(@(t,q) flightRhs(t,q,c),[0 t0],q0,opts);
        q0=qq(end,:)';
    end
    opts=odeset(opts,'Events',@(t,q) flightEvents(t,q,c));
    [~,~,te,qe,ie]=ode113(@(t,q) flightRhs(t,q,c),[t0 10],q0,opts);
    if isempty(ie), return; end
    if ie(1)==1
        status='crossed'; result=[te(1),qe(1,2),qe(1,3)];
    else
        status='grounded';
    end
end

function dq=flightRhs(~,q,c)
    v=q(4:6); u=v-c.wind(:);
    factor=c.constants.rho*pi*c.constants.R^2/(2*c.constants.m);
    acceleration=[0;0;-c.constants.g]+factor*norm(u)*(-c.cd*u+cross(c.b(:),u));
    dq=[v;acceleration];
end

function [value,isterminal,direction]=flightEvents(~,q,c)
    value=[q(1)-c.target_x;q(3)-c.constants.R];
    isterminal=[1;1]; direction=[1;-1];
end
