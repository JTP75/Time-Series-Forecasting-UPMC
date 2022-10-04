function [net,arch,opts] = loadnet(name)

skip_errs = true;
warn_on_net = false;


d = pwd + "/mdl";
archfile = fullfile(d,name,"architecture.mat");
optsfile = fullfile(d,name,"options.mat");
ntwkfile = fullfile(d,name,"network.mat");

% load network
try
    net_struct = load(ntwkfile);
    net = net_struct.net;
catch ME
    msg = "Error loading '"+name+"/network.mat'.";
    id = "loadnet:fileLoadError";
    if ~exist("mdl/"+name,"dir")
        msg = msg+" Directory '"+name+"' not found.";
        id = "loadnet:dirNotFoundError";
    elseif ~exist("mdl/"+name+"/network.mat","file")
        msg = msg+" File 'network.mat' not found in '"+name+"' directory.";
        id = "loadnet:fileNotFoundError";
    end
    CE = MException(id,msg);
    ME = addCause(ME,CE);
    if skip_errs
        warning('%s\n\nError in %s (%s) (line %d)\n', ME.message,...
            ME.stack(1).('name'), ME.stack(1).('file'), ME.stack(1).('line'));
        net = [];
    else
        rethrow(ME); %#ok<UNRCH>
    end
end


% load architecture
try
    arch_struct = load(archfile);
    arch = arch_struct.layers;
catch ME
    msg = "Error loading '"+name+"/architecture.mat'.";
    id = "loadnet:fileLoadError";
    if ~exist("mdl/"+name,"dir")
        msg = msg+" Directory '"+name+"' not found.";
        id = "loadnet:dirNotFoundError";
    elseif ~exist("mdl/"+name+"/architecture.mat","file")
        msg = msg+" File 'architecture.mat' not found in '"+name+"' directory.";
        id = "loadnet:fileNotFoundError";
    end
    CE = MException(id,msg);
    ME = addCause(ME,CE); %#ok<NASGU>
    if skip_errs
        if warn_on_net
            warning('%s\n\nError in %s (%s) (line %d)\n', ME.message,...
                ME.stack(1).('name'), ME.stack(1).('file'), ME.stack(1).('line')); %#ok<UNRCH>
        end
        arch = [];
    else
        rethrow(ME); %#ok<UNRCH>
    end
end


% load options
try
    opts_struct = load(optsfile);
    opts = opts_struct.opts;
catch ME
    msg = "Error loading '"+name+"\\options.mat'.";
    id = "loadnet:fileLoadError";
    if ~exist("mdl/"+name,"dir")
        msg = msg+" Directory '"+name+"' not found.";
        id = "loadnet:dirNotFoundError";
    elseif ~exist("mdl/"+name+"/options.mat","file")
        msg = msg+" File 'options.mat' not found in '"+name+"' directory.";
        id = "loadnet:fileNotFoundError";
    end
    CE = MException(id,msg);
    ME = addCause(ME,CE);
    if skip_errs
        warning('%s\n\nError in %s (%s) (line %d)\n', ME.message,...
            ME.stack(1).('name'), ME.stack(1).('file'), ME.stack(1).('line'));
        opts = [];
    else
        rethrow(ME); %#ok<UNRCH>
    end
end





